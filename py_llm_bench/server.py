"""SGLang server lifecycle management.

Handles starting the sglang server inside the container, health checking,
warmup requests, and stopping.
"""

import shlex
import sys
import threading
import time

import httpx

from .config import SuiteConfig
from .container import ContainerManager


class ServerError(Exception):
    """Base exception for server errors."""


class ServerCrashError(ServerError):
    """Server process or container exited unexpectedly."""


class ServerTimeoutError(ServerError):
    """Server did not become healthy within timeout."""


class ServerManager:
    """Manages the SGLang server lifecycle inside a Docker container."""

    def __init__(self, container: ContainerManager, config: SuiteConfig):
        self.container = container
        self.config = config
        self._log_thread: threading.Thread | None = None
        self._log_stop = threading.Event()

    def build_server_cmd(self, skip_warmup: bool = False) -> list[str]:
        """Build the sglang server launch command."""
        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", self.config.model_path,
            "--host", "0.0.0.0",
            "--port", str(self.config.port),
        ]
        cmd.extend(self.config.server_args)
        cmd.extend(["--watchdog-timeout", str(self.config.watchdog_timeout)])
        cmd.extend(["--soft-watchdog-timeout", str(self.config.watchdog_timeout)])
        if skip_warmup:
            cmd.append("--skip-server-warmup")
        return cmd

    def start(self, skip_warmup: bool = False) -> None:
        """Start the sglang server in the background inside the container."""
        cmd = self.build_server_cmd(skip_warmup)
        shell_cmd = " ".join(shlex.quote(p) for p in cmd) + " > /tmp/server.log 2>&1"
        print(f">>> Starting server: {' '.join(cmd)}")

        # Build container env for the exec
        env = {}
        for spec in self.config.container_env_overrides:
            k, v = spec.split("=", 1)
            env[k] = v

        self.container.exec_run(
            ["bash", "-c", shell_cmd],
            environment=env if env else None,
            detach=True,
        )

    def wait_healthy(self, timeout: int | None = None) -> None:
        """Wait for the server to become healthy.

        Polls GET /health every 5 seconds. Requires `stable_required` consecutive
        successes. Checks that the container is still running each iteration.

        Args:
            timeout: Max polling iterations (each 5s). Defaults to config.health_timeout.

        Raises:
            ServerCrashError: If the container disappears.
            ServerTimeoutError: If timeout expires.
        """
        timeout = timeout or self.config.health_timeout
        max_seconds = timeout * 5
        stable_required = 3
        stable_count = 0

        print(">>> Waiting for server to become healthy...")
        self._start_log_tail()

        deadline = time.monotonic() + max_seconds
        while time.monotonic() < deadline:
            if not self.container.is_running():
                self._stop_log_tail()
                raise ServerCrashError(
                    f"Container '{self.container.container_name}' no longer running. "
                    "Server may have crashed."
                )

            try:
                resp = httpx.get(
                    f"http://localhost:{self.config.port}/health",
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    stable_count += 1
                    if stable_count >= stable_required:
                        self._stop_log_tail()
                        elapsed = max_seconds - (deadline - time.monotonic())
                        print(f"\n>>> Server is ready! (stable after {elapsed:.0f}s, "
                              f"{stable_count} consecutive checks)")
                        return
                else:
                    if stable_count > 0:
                        print(f"\n>>> Server health unstable (was healthy {stable_count}x, "
                              "then failed). Resetting...")
                    stable_count = 0
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
                if stable_count > 0:
                    print(f"\n>>> Server health unstable (was healthy {stable_count}x, "
                          "then failed). Resetting...")
                stable_count = 0

            time.sleep(5)

        self._stop_log_tail()
        raise ServerTimeoutError(f"Server failed to start within {max_seconds}s")

    def warmup(self) -> None:
        """Send a warmup request to trigger JIT compilation, then re-check health."""
        print(">>> Sending warmup request to trigger JIT compilation...")
        try:
            httpx.post(
                f"http://localhost:{self.config.port}/v1/completions",
                json={
                    "model": self.config.model_path,
                    "prompt": "hello",
                    "max_tokens": 1,
                },
                timeout=120.0,
            )
        except Exception:
            pass  # Warmup may fail during JIT, that's OK

        print(">>> Waiting for server to stabilize after warmup...")
        time.sleep(5)
        self.wait_healthy()

    def stop(self) -> None:
        """Kill the sglang server process inside the container."""
        try:
            self.container.exec_run(["pkill", "-f", "sglang.launch_server"])
        except Exception:
            pass

    def describe(self, skip_warmup: bool = False) -> dict:
        """Return a description of the server config (for dry-run)."""
        cmd = self.build_server_cmd(skip_warmup)
        return {
            "server_command": " ".join(cmd),
            "port": self.config.port,
            "health_timeout": f"{self.config.health_timeout * 5}s ({self.config.health_timeout} polls x 5s)",
            "watchdog_timeout": f"{self.config.watchdog_timeout}s",
            "stability_checks": 3,
        }

    def _start_log_tail(self) -> None:
        """Start tailing /tmp/server.log in a background thread."""
        self._log_stop.clear()

        def _tail():
            try:
                exec_id, stream = self.container.exec_run(
                    ["bash", "-c", "tail -f /tmp/server.log 2>/dev/null"],
                    stream=True,
                )
                for chunk in stream:
                    if self._log_stop.is_set():
                        break
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
            except Exception:
                pass

        self._log_thread = threading.Thread(target=_tail, daemon=True)
        self._log_thread.start()

    def _stop_log_tail(self) -> None:
        """Stop the background log tail."""
        self._log_stop.set()
        if self._log_thread and self._log_thread.is_alive():
            self._log_thread.join(timeout=3)
        self._log_thread = None
