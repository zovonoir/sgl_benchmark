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
    """Manages Docker container lifecycle for a benchmark run."""

    def __init__(self, config: SuiteConfig, run_dir: Path, script_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.script_dir = script_dir
        _rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        self.container_name = f"llm_bench_{getpass.getuser()}_{os.getpid()}_{_rand}"
        self._client = docker.from_env()
        self._container = None

    def start(self) -> None:
        """Start a new Docker container with all configured mounts and env vars."""
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

        # Environment variables
        env = {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "HF_HOME": "/.cache/huggingface/",
        }

        # container_env: all user-specified env vars (format: "KEY=VALUE")
        for spec in self.config.container_env:
            if "=" in spec:
                k, v = spec.split("=", 1)
                env[k] = v

        try:
            # Base docker run kwargs
            run_kwargs = dict(
                image=self.config.image,
                name=self.container_name,
                detach=True,
                stdin_open=True,
                tty=True,
                user="root",
                cap_add=["SYS_PTRACE"],
                security_opt=["seccomp=unconfined"],
                devices=["/dev/kfd", "/dev/dri"],
                group_add=["video"],
                ipc_mode="host",
                pid_mode="host",
                network_mode="host",
                privileged=True,
                environment=env,
                mounts=mounts,
            )

            # Apply user-specified docker run args (shm_size, ulimits, etc.)
            for k, v in self.config.docker_run_args.items():
                # Handle ulimits specially: convert dict to docker Ulimit objects
                if k == "ulimits":
                    import docker.types
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
        """Stop and remove the container."""
        if self._container is None:
            return

        try:
            # Try to kill the sglang server first
            self._container.exec_run(
                ["pkill", "-f", "sglang.launch_server"],
            )
            time.sleep(2)
        except (APIError, NotFound):
            pass

        try:
            self._container.remove(force=True)
        except (APIError, NotFound):
            pass

        self._container = None

    def cleanup_stale(self) -> None:
        """Remove stale containers from previous crashed runs.

        Only removes containers whose originating PID no longer exists.
        """
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
            exit_code, output = self.exec_run(["bash", "-c", cmd])
            if output:
                text = output.decode("utf-8", errors="replace").strip()
                if text:
                    print(f"    {text}")
            if exit_code != 0:
                raise ContainerError(f"Post-start command failed (exit {exit_code}): {cmd}")

    def describe(self) -> dict:
        """Return a description of what this container manager would do (for dry-run)."""
        env = {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "HF_HOME": "/.cache/huggingface/",
        }
        for spec in self.config.container_env:
            if "=" in spec:
                k, v = spec.split("=", 1)
                env[k] = v

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
            "image": self.config.image,
            "container_name": self.container_name,
            "environment": env,
            "mounts": mounts,
            "devices": ["/dev/kfd", "/dev/dri"],
            "network_mode": "host",
            "privileged": True,
            "docker_run_args": self.config.docker_run_args if self.config.docker_run_args else None,
            "post_start_commands": self.config.post_start_commands,
        }
