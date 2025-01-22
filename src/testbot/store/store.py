from abc import ABC, abstractmethod
from typing import Any, Optional
from pathlib import Path

from .models import RepoConfig, TestFileData

# TODO: consider adding a user parameter to support serverside?
class TestBotStore(ABC):
    def create_repo(self, repo_config: RepoConfig) -> Optional[Any]:
        pass
    
    def get_repo_config(self, repo_name: str) -> RepoConfig:
        pass

    def update_or_create_testfile_data(self, tf_data: TestFileData) -> Optional[Any]:
        pass

    def get_testfile_data(self, tf_path: Path) -> TestFileData:
        pass


