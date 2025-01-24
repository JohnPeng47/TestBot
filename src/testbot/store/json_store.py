import json
from pathlib import Path
from typing import Optional, Any

from testbot.config import STORE_PATH
from .store import TestBotStore
from ..models.core import RepoConfig, TestFileData

class JsonStore(TestBotStore):
    """JSON file-based implementation of BaseStore"""
    
    def __init__(self, store_path: Path = STORE_PATH):
        self.store_path = store_path
        self.repos_file = store_path / "repos.json"
        self.tests_file = store_path / "tests.json"
        
        # Create store files if they don't exist
        self.store_path.mkdir(parents=True, exist_ok=True)
        if not self.repos_file.exists():
            self._write_json(self.repos_file, {})
        if not self.tests_file.exists():
            self._write_json(self.tests_file, {})

    def create_repoconfig(self, repo_config: RepoConfig) -> Optional[Any]:
        repos = self._read_json(self.repos_file)
        # overwrite old config if already exists
        repos[repo_config.repo_name] = repo_config.dict()

        print("Saving to repo: ", repo_config.repo_name)
        self._write_json(self.repos_file, repos)

        return repo_config
    
    def delete_repoconfig(self, repo_name: str) -> None:
        repos = self._read_json(self.repos_file)
        if repo_name not in repos:
            raise KeyError(f"Repository {repo_name} not found")
        del repos[repo_name]
        self._write_json(self.repos_file, repos)

    def get_repo_config(self, repo_name: str) -> RepoConfig:
        repos = self._read_json(self.repos_file)
        if repo_name not in repos:
            raise KeyError(f"Repository {repo_name} not found")
        return RepoConfig(**repos[repo_name])

    def update_or_create_testfile_data(self, tf_data: TestFileData) -> Optional[Any]:
        tests = self._read_json(self.tests_file)
        tests[tf_data.filepath] = tf_data.dict()
        self._write_json(self.tests_file, tests)
        return tf_data

    def get_testfiles_from_srcfile(self, source_file: Path) -> Optional[TestFileData]:
        tests = self._read_json(self.tests_file)
        source_file_str = str(source_file)
        
        # Look for test files that map to this source file
        for _, test_data in tests.items():
            if test_data.get("source_file") == source_file_str:
                return TestFileData(**test_data)
                
        return None

    def _read_json(self, file_path: Path) -> dict:
        with open(file_path, "r") as f:
            return json.load(f)

    def _write_json(self, file_path: Path, data: dict):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
