"""Base runner with shared lifecycle for all test modes."""

from abc import ABC, abstractmethod
from pathlib import Path

from ..config import SuiteConfig
from ..container import ContainerManager
from ..server import ServerManager


class BaseRunner(ABC):
    """Template method pattern for all test modes.

    Subclasses implement `execute()` for mode-specific logic and
    `dry_run()` for dry-run output.
    """

    def __init__(self, config: SuiteConfig, run_dir: Path, script_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.script_dir = script_dir
        self.container = ContainerManager(config, run_dir, script_dir)
        self.server = ServerManager(self.container, config)

    def run(self) -> None:
        """Execute the full lifecycle: setup -> execute -> teardown.

        Subclasses call container.start() and container.run_post_start_commands()
        in their execute() method at the appropriate time.
        """
        try:
            self.execute()
        finally:
            self.container.cleanup()

    @abstractmethod
    def execute(self) -> None:
        """Mode-specific logic. Subclasses implement this."""

    @abstractmethod
    def dry_run(self) -> None:
        """Print what this runner would do without actually running."""
