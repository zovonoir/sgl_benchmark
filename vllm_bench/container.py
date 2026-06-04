"""Docker helpers for vLLM benchmark runs."""

from __future__ import annotations

import io
import os
import random
import string
import tarfile
from pathlib import Path
from typing import Iterable

import docker
import docker.types
from docker.errors import APIError, NotFound

from .config import SuiteConfig


class ContainerError(Exception):
    """Docker operation failed."""


class ExistingContainerManager:
    """Attach to an existing container or create one from an image."""

    def __init__(self, config: SuiteConfig, project_root: Path, run_dir: Path):
        self.config = config
        self.project_root = project_root
        self.run_dir = run_dir
        self.create_mode = bool(config.image)
        if config.existing_container:
            self.container_name = config.existing_container
        else:
            rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
            self.container_name = config.container_name or f"vllm_bench_{os.getpid()}_{rand}"
        self.suite_path = config.suite_path_in_container
        self.output_path = f"{self.suite_path}/_output"
        self._client = docker.from_env()
        self._container = None
        self._env = config.container_environment()

    def attach(self) -> None:
        if self.create_mode:
            self._start_create()
            return

        try:
            self._container = self._client.containers.get(self.container_name)
        except NotFound as exc:
            raise ContainerError(f"Container not found: {self.container_name}") from exc
        except APIError as exc:
            raise ContainerError(f"Failed to inspect container: {exc}") from exc

        self._container.reload()
        if self._container.status != "running":
            raise ContainerError(
                f"Container '{self.container_name}' is not running "
                f"(status: {self._container.status})"
            )
        print(f">>> Attached to existing container: {self.container_name}")

    def _start_create(self) -> None:
        """Create a detached container that stays alive for docker exec calls."""

        env = dict(self._env)
        mounts = self._build_mounts()
        run_kwargs = {
            "image": self.config.image,
            "name": self.container_name,
            "detach": True,
            "stdin_open": True,
            "tty": True,
            "environment": env,
            "mounts": mounts,
        }
        if self.config.entrypoint:
            run_kwargs["entrypoint"] = self.config.entrypoint
        run_kwargs["command"] = self.config.command or ["-lc", "while true; do sleep 3600; done"]

        for key, value in self.config.docker_run_args.items():
            if key == "ulimits":
                run_kwargs["ulimits"] = self._build_ulimits(value)
            else:
                run_kwargs[key] = value

        try:
            self._container = self._client.containers.run(**run_kwargs)
        except APIError as exc:
            raise ContainerError(f"Failed to start container from image {self.config.image}: {exc}") from exc
        print(f">>> Started container {self.container_name} from image: {self.config.image}")

    def _build_mounts(self) -> list[docker.types.Mount]:
        extra_targets = {
            _normalize_mount_target(spec.split(":", 2)[1])
            for spec in self.config.extra_container_mounts
            if ":" in spec
        }
        mounts = []
        auto_mounts = [
            (self.config.host_model_mount_path, "/.cache/huggingface/"),
            (self.config.host_model_mount_path, self.config.host_model_mount_path),
        ]
        for src, dst in auto_mounts:
            if _normalize_mount_target(dst) not in extra_targets:
                mounts.append(docker.types.Mount(target=dst, source=src, type="bind"))
        for spec in self.config.extra_container_mounts:
            parts = spec.split(":")
            if len(parts) < 2:
                raise ContainerError(f"Invalid mount spec: {spec}")
            src, dst = parts[0], parts[1]
            read_only = len(parts) > 2 and parts[2] == "ro"
            mounts.append(docker.types.Mount(target=dst, source=src, type="bind", read_only=read_only))
        return mounts

    @staticmethod
    def _build_ulimits(config: dict) -> list[docker.types.Ulimit]:
        ulimits = []
        for name, value in config.items():
            if isinstance(value, dict):
                ulimits.append(docker.types.Ulimit(
                    name=name,
                    soft=value.get("soft", 0),
                    hard=value.get("hard", 0),
                ))
            else:
                ulimits.append(docker.types.Ulimit(name=name, soft=value, hard=value))
        return ulimits

    def exec_run(
        self,
        cmd: list[str],
        *,
        environment: dict[str, str] | None = None,
        workdir: str | None = None,
        stream: bool = False,
    ):
        if self._container is None:
            raise ContainerError("Container not attached")

        merged_env = dict(self._env)
        if environment:
            merged_env.update(environment)
        merged_env = {str(key): "" if value is None else str(value) for key, value in merged_env.items()}
        cmd = [str(part) for part in cmd]

        try:
            if stream:
                exec_id = self._client.api.exec_create(
                    self._container.id,
                    cmd,
                    environment=merged_env,
                    workdir=workdir,
                )
                output = self._client.api.exec_start(exec_id["Id"], stream=True)
                return exec_id["Id"], output

            result = self._container.exec_run(
                cmd,
                environment=merged_env,
                workdir=workdir,
            )
            return result.exit_code, result.output
        except APIError as exc:
            raise ContainerError(f"exec_run failed: {exc}") from exc

    def inspect_exec(self, exec_id: str) -> dict:
        return self._client.api.exec_inspect(exec_id)

    def run_commands(self, commands: list[str], label: str) -> None:
        """Run setup commands inside the container, streaming output."""

        if not commands:
            return
        print(f">>> Running {len(commands)} {label} command(s)...")
        for command in commands:
            print(f"    $ {command}")
            exec_id, output_stream = self.exec_run(["bash", "-lc", command], stream=True)
            for chunk in output_stream:
                import sys

                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            exit_info = self.inspect_exec(exec_id)
            exit_code = int(exit_info.get("ExitCode", -1))
            if exit_code != 0:
                raise ContainerError(f"{label} command failed with exit {exit_code}: {command}")

    def cleanup_container(self) -> None:
        """Remove containers created by vllm_bench; never remove attached containers."""

        if not self.create_mode or self._container is None:
            self._container = None
            return
        try:
            self._container.remove(force=True)
        except (APIError, NotFound):
            pass
        self._container = None

    def inject_suite(self) -> None:
        """Copy the project root into the configured container suite path."""

        if self._container is None:
            raise ContainerError("Container not attached")

        parent = os.path.dirname(self.suite_path.rstrip("/")) or "/"
        basename = os.path.basename(self.suite_path.rstrip("/"))
        self.exec_run(["bash", "-lc", f"rm -rf {quote(self.suite_path)} && mkdir -p {quote(parent)}"])

        tar_bytes = self._build_project_tar(basename)
        try:
            self._container.put_archive(parent, tar_bytes)
        except APIError as exc:
            raise ContainerError(f"Failed to copy suite into container: {exc}") from exc
        print(f">>> Injected benchmark suite into container: {self.suite_path}")

    def copy_case_results(self, case_name: str, host_case_dir: Path) -> None:
        """Copy a case output directory from the container to the host."""

        if self._container is None:
            raise ContainerError("Container not attached")

        host_case_dir.mkdir(parents=True, exist_ok=True)
        container_case_dir = f"{self.output_path}/{case_name}"
        try:
            stream, _ = self._container.get_archive(container_case_dir)
        except APIError as exc:
            raise ContainerError(f"Failed to copy case results: {exc}") from exc

        data = io.BytesIO(b"".join(stream))
        with tarfile.open(fileobj=data, mode="r") as tar:
            self._safe_extract_tar(tar, host_case_dir.parent)
        copied = host_case_dir.parent / case_name
        if copied != host_case_dir and copied.exists():
            # Docker archives the directory itself; normally this branch is not used.
            pass
        print(f">>> Copied results to {host_case_dir}")

    def _build_project_tar(self, suite_basename: str) -> bytes:
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as tar:
            for path in self._iter_project_files():
                rel = path.relative_to(self.project_root)
                arcname = str(Path(suite_basename) / rel)
                tar.add(path, arcname=arcname, recursive=False)
        data.seek(0)
        return data.getvalue()

    def _iter_project_files(self) -> Iterable[Path]:
        excluded_dirs = {".git", "__pycache__", "runs", ".pytest_cache"}
        excluded_suffixes = {".pyc", ".pyo"}
        for path in self.project_root.rglob("*"):
            rel_parts = path.relative_to(self.project_root).parts
            if any(part in excluded_dirs for part in rel_parts):
                continue
            if path.suffix in excluded_suffixes:
                continue
            if path.is_file():
                yield path

    @staticmethod
    def _safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
        dest = dest.resolve()
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest)):
                raise ContainerError(f"Unsafe path in archive: {member.name}")
        tar.extractall(dest)


def _normalize_mount_target(value: str) -> str:
    return value.rstrip("/") or "/"


def quote(value: str) -> str:
    """Small shell quoting helper to avoid importing shlex at call sites."""

    import shlex

    return shlex.quote(value)

