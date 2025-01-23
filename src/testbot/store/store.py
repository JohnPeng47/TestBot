from abc import ABC, abstractmethod
from typing import Any, Optional
from pathlib import Path

from ..models.core import RepoConfig, TestFileData


# TODO: consider adding a user parameter to support serverside?
class TestBotStore(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_repoconfig(self, repo_config: RepoConfig) -> Optional[Any]:
        pass
    
    @abstractmethod
    def get_repo_config(self, repo_name: str) -> RepoConfig:
        pass

    @abstractmethod
    def delete_repoconfig(self, repo_name: str) -> None:
        pass

    @abstractmethod
    def update_or_create_testfile_data(self, tf_data: TestFileData) -> Optional[Any]:
        pass

    @abstractmethod
    def get_testfile_data(self, tf_path: Path) -> TestFileData:
        pass


