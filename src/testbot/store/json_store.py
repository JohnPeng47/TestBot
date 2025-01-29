import json
from pathlib import Path
from typing import Optional, Any, List

from testbot.config import STORE_PATH
from testbot.utils import hook_print

from .store import TestBotStore
from .exceptions import StoreDoesNotExist
from ..models.core import RepoConfig, TestFileData

class JsonStore(TestBotStore):
    """JSON file-based implementation of BaseStore"""
    
    def __init__(self, store_path: Path = STORE_PATH):
        self.store_path = store_path
        self.repos_file = store_path / "repos.json"
        self.tests_file = store_path / "tests.json"

        hook_print(f"[Store path]: {self.tests_file}")
        hook_print(f"[Repos file]: {self.repos_file}")
        
        self.store_path.mkdir(parents=True, exist_ok=True)
        if not self.repos_file.exists():
            self._write_json(self.repos_file, {})
        if not self.tests_file.exists():
            self._write_json(self.tests_file, {})

    def create_repoconfig(self, repo_config: RepoConfig) -> Optional[Any]:
        repos = self._read_json(self.repos_file)
        # overwrite old config if already exists
        repos[repo_config.repo_name] = repo_config

        print("Saving to repo: ", repo_config.repo_name)
        self._write_json(self.repos_file, repos)

        return repo_config
    
    def delete_repoconfig(self, repo_name: str) -> None:
        repos = self._read_json(self.repos_file)
        if repo_name not in repos:
            raise KeyError(f"Repository {repo_name} not found")

        del repos[repo_name]
        self._write_json(self.repos_file, repos)

    def get_repoconfig(self, filter_fn) -> RepoConfig:
        try:
            for repo in self._read_json(self.repos_file).values():
                if filter_fn(repo):
                    return repo
        except StopIteration:
            return None
        
    def update_or_create_testfile_data(self, tf_data: TestFileData) -> Optional[Any]:
        tests = self._read_json(self.tests_file)
        tests[tf_data.filepath] = tf_data
        self._write_json(self.tests_file, tests)
        return tf_data
    
    def get_testfiles_from_srcfile(self, source_file: Path) -> List[TestFileData]:
        tests = self._read_json(self.tests_file)
        source_file_str = str(source_file)
        
        # Look for test files that have this source file in their targeted files
        matching_tests = []
        for _, test_data in tests.items():
            if source_file_str in test_data.targeted_files:
                matching_tests.append(test_data)
                
        return matching_tests
    
    def _read_json(self, file_path: Path) -> dict[str, RepoConfig] | dict[str, TestFileData]:
        with open(file_path, "r") as f:
            data = json.load(f)
            if file_path == self.repos_file:
                return {k: RepoConfig(**v) for k, v in data.items()}
            else:
                return {k: TestFileData(**v) for k, v in data.items()}

    def _write_json(self, file_path: Path, data: dict[str, RepoConfig] | dict[str, TestFileData]):
        with open(file_path, "w") as f:
            json_data = {k: v.dict() for k, v in data.items()}
            json.dump(json_data, f, indent=2)
