"""Docker container lifecycle management for SGLang Benchmark Suite.

Wraps the Docker Python SDK to create, exec into, and remove containers.
Replaces all docker run/exec/rm shell commands from the bash version.
"""

import getpass
import os
import random
import string
import sys
import time
from pathlib import Path

import docker
import docker.types
from docker.errors import NotFound, APIError

from .config import SuiteConfig


class ContainerError(Exception):
    """Docker container lifecycle failure."""


class ContainerManager:
    """Manages Docker container lifecycle for a benchmark run.

    Two modes:
    - Create mode (config.image set): creates a new container via docker run
    - Attach mode (config.existing_container set): connects to an existing container
    """

    def __init__(self, config: SuiteConfig, run_dir: Path, script_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.script_dir = script_dir
        self.attach_mode = bool(config.existing_container)
        if self.attach_mode:
            self.container_name = config.existing_container
        else:
            _rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            self.container_name = f"llm_bench_{getpass.getuser()}_{os.getpid()}_{_rand}"
        self._client = docker.from_env()
        self._container = None
        self._extra_start_env = None  # stored for exec_run injection in attach mode

    def start(self, extra_env: dict | None = None) -> None:
        """Start or attach to a Docker container.

        In create mode: creates a new container with all configured mounts and env vars.
        In attach mode: connects to the existing container and stores env vars for
        injection via docker exec.

        Args:
            extra_env: Additional environment variables to inject (e.g. from runner).
        """
        if self.attach_mode:
            self._start_attach(extra_env)
            return

        self._start_create(extra_env)

    def _start_attach(self, extra_env: dict | None = None) -> None:
        """Attach to an existing container."""
        try:
            self._container = self._client.containers.get(self.container_name)
        except NotFound:
            raise ContainerError(
                f"Container '{self.container_name}' not found. "
                "Make sure it exists and is running."
            )
        except APIError as e:
            raise ContainerError(f"Failed to connect to container: {e}") from e

        if self._container.status != "running":
            raise ContainerError(
                f"Container '{self.container_name}' is not running "
                f"(status: {self._container.status}). Start it first."
            )

        # Build env vars to inject via docker exec
        self._extra_start_env = {}
        for spec in self.config.container_env:
            if "=" in spec:
                k, v = spec.split("=", 1)
                self._extra_start_env[k] = v
        if extra_env:
            self._extra_start_env.update(extra_env)

        print(f">>> Attached to existing container: {self.container_name}")
        print(f">>> Environment variables will be injected via docker exec")

    def _start_create(self, extra_env: dict | None = None) -> None:
        """Create a new container."""
        mounts = [
            docker.types.Mount(
                target="/.cache/huggingface/",
                source=self.config.host_model_mount_path,
                type="bind",
            ),
            docker.types.Mount(
                target=self.config.host_model_mount_path,
                source=self.config.host_model_mount_path,
                type="bind",
            ),
            docker.types.Mount(
                target="/simple-suite",
                source=str(self.script_dir),
                type="bind",
            ),
            docker.types.Mount(
                target="/simple-suite-output",
                source=str(self.run_dir),
                type="bind",
            ),
            docker.types.Mount(
                target="/dev/shm",
                source="/dev/shm",
                type="bind",
            ),
        ]

        # Extra container mounts from config (format: "src:dst" or "src:dst:ro")
        for mount_spec in self.config.extra_container_mounts:
            parts = mount_spec.split(":")
            src, dst = parts[0], parts[1]
            read_only = len(parts) > 2 and parts[2] == "ro"
            mounts.append(docker.types.Mount(
                target=dst, source=src, type="bind", read_only=read_only,
            ))

        # Environment variables (all from user config, no hardcoded defaults)
        env = {}

        # container_env: all user-specified env vars (format: "KEY=VALUE")
        for spec in self.config.container_env:
            if "=" in spec:
                k, v = spec.split("=", 1)
                env[k] = v

        # Extra env from runner (e.g. SGLANG_TORCH_PROFILER_DIR for profile mode)
        if extra_env:
            env.update(extra_env)

        try:
            # Only image, name, detach, tty, environment, mounts are automatic.
            # All other docker run parameters come from config.docker_run_args.
            run_kwargs = dict(
                image=self.config.image,
                name=self.container_name,
                detach=True,
                stdin_open=True,
                tty=True,
                environment=env,
                mounts=mounts,
            )

            # Apply all user-specified docker run args
            for k, v in self.config.docker_run_args.items():
                if k == "ulimits":
                    ulimit_list = []
                    for ul_name, ul_val in v.items():
                        if isinstance(ul_val, dict):
                            ulimit_list.append(docker.types.Ulimit(
                                name=ul_name, soft=ul_val.get("soft", 0), hard=ul_val.get("hard", 0),
                            ))
                        else:
                            ulimit_list.append(docker.types.Ulimit(
                                name=ul_name, soft=ul_val, hard=ul_val,
                            ))
                    run_kwargs["ulimits"] = ulimit_list
                else:
                    run_kwargs[k] = v

            self._container = self._client.containers.run(**run_kwargs)
        except APIError as e:
            raise ContainerError(f"Failed to start container: {e}") from e

    def exec_run(self, cmd: list[str], environment: dict | None = None,
                 workdir: str | None = None, stream: bool = False,
                 detach: bool = False):
        """Execute a command inside the container.

        Args:
            cmd: Command to execute as a list of strings.
            environment: Additional environment variables.
            workdir: Working directory inside the container.
            stream: If True, returns (exit_code, output_generator).
            detach: If True, runs the command in the background.

        Returns:
            If stream=False and detach=False: (exit_code, output_bytes)
            If stream=True: (exit_code_or_None, output_generator)
            If detach=True: exec_id
        """
        if self._container is None:
            raise ContainerError("Container not started")

        # In attach mode, merge stored env vars into every exec call
        if self.attach_mode and self._extra_start_env:
            merged_env = dict(self._extra_start_env)
            if environment:
                merged_env.update(environment)
            environment = merged_env

        try:
            if detach:
                return self._container.exec_run(
                    cmd, environment=environment, workdir=workdir,
                    detach=True,
                )

            if stream:
                exec_id = self._client.api.exec_create(
                    self._container.id, cmd,
                    environment=environment, workdir=workdir,
                )
                output = self._client.api.exec_start(exec_id["Id"], stream=True)
                return exec_id["Id"], output

            result = self._container.exec_run(
                cmd, environment=environment, workdir=workdir,
            )
            return result.exit_code, result.output

        except APIError as e:
            raise ContainerError(f"exec_run failed: {e}") from e

    def is_running(self) -> bool:
        """Check if the container is still running."""
        if self._container is None:
            return False
        try:
            self._container.reload()
            return self._container.status == "running"
        except NotFound:
            return False
        except APIError:
            return False

    def cleanup(self) -> None:
        """Clean up after test.

        In create mode: remove the container.
        In attach mode: NEVER remove the container — only release the reference.
        """
        if self._container is None:
            return

        if self.attach_mode:
            # Attach mode: absolutely do NOT remove the user's container
            self._container = None
            return

        # Create mode: safe to remove our own container
        try:
            self._container.remove(force=True)
        except (APIError, NotFound):
            pass

        self._container = None

    def cleanup_stale(self) -> None:
        """Remove stale containers from previous crashed runs.

        Only removes containers whose originating PID no longer exists.
        Skipped entirely in attach mode.
        """
        if self.attach_mode:
            return
        user = getpass.getuser()
        prefix = f"llm_bench_{user}_"

        try:
            containers = self._client.containers.list(all=True)
        except APIError:
            return

        for c in containers:
            name = c.name
            if not name.startswith(prefix) or name == self.container_name:
                continue

            # Extract PID from container name (format: llm_bench_{user}_{pid}_{rand})
            suffix = name[len(prefix):]
            pid_str = suffix.split("_")[0]
            if not pid_str.isdigit():
                continue

            pid = int(pid_str)
            try:
                os.kill(pid, 0)
                # Process still exists, skip
            except OSError:
                # Process is gone, remove the stale container
                print(f">>> Removing stale container (PID {pid} no longer exists): {name}")
                try:
                    c.remove(force=True)
                except (APIError, NotFound):
                    pass

    def run_post_start_commands(self) -> None:
        """Run post_start_commands inside the container.

        These commands execute after container start and before server start,
        allowing users to inject custom configurations (e.g., tuned GEMM CSVs,
        environment setup, file modifications).
        """
        if not self.config.post_start_commands:
            return

        print(f">>> Running {len(self.config.post_start_commands)} post-start command(s)...")
        for cmd in self.config.post_start_commands:
            print(f"    $ {cmd}")
            exec_id, output_stream = self.exec_run(
                ["bash", "-c", cmd], stream=True,
            )
            for chunk in output_stream:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            exit_info = self._client.api.exec_inspect(exec_id)
            exit_code = exit_info.get("ExitCode", -1)
            if exit_code != 0:
                raise ContainerError(f"Post-start command failed (exit {exit_code}): {cmd}")

    def describe(self, extra_env: dict | None = None) -> dict:
        """Return a description of what this container manager would do (for dry-run)."""
        env = {}
        for spec in self.config.container_env:
            if "=" in spec:
                k, v = spec.split("=", 1)
                env[k] = v
        if extra_env:
            env.update(extra_env)

        if self.attach_mode:
            return {
                "mode": "attach",
                "container_name": self.container_name,
                "suite_path": self.config.suite_path_in_container,
                "environment": env,
                "env_injection": "docker exec",
                "post_start_commands": self.config.post_start_commands,
            }

        mounts = [
            f"{self.config.host_model_mount_path} -> /.cache/huggingface/",
            f"{self.config.host_model_mount_path} -> {self.config.host_model_mount_path}",
            f"{self.script_dir} -> /simple-suite",
            f"{self.run_dir} -> /simple-suite-output",
            "/dev/shm -> /dev/shm",
        ]
        for spec in self.config.extra_container_mounts:
            parts = spec.split(":")
            mounts.append(f"{parts[0]} -> {parts[1]}" + (f" ({parts[2]})" if len(parts) > 2 else ""))

        return {
            "mode": "create",
            "image": self.config.image,
            "container_name": self.container_name,
            "environment": env,
            "mounts": mounts,
            "docker_run_args": self.config.docker_run_args if self.config.docker_run_args else None,
            "post_start_commands": self.config.post_start_commands,
        }
