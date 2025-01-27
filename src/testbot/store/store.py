from abc import ABC, abstractmethod
from typing import Any, Optional, Callable
from pathlib import Path

from ..models.core import RepoConfig, TestFileData

class TestBotStore(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def create_repoconfig(self, repo_config: RepoConfig) -> Any | None:
        pass
    
    @abstractmethod
    def get_repoconfig(self, filter_fn: Callable[[RepoConfig], bool]) -> RepoConfig | None:
        pass

    @abstractmethod
    def delete_repoconfig(self, repo_name: str) -> None:
        pass

    @abstractmethod
    def update_or_create_testfile_data(self, tf_data: TestFileData) -> Any | None:
        pass

    @abstractmethod
    def get_testfiles_from_srcfile(self, source_file: Path) -> TestFileData | None:
        pass