from pytest import fixture
import git

from pathlib import Path
import shutil
from testbot.store.json_store import JsonStore

TEST_REPO = Path(__file__).parent / "test_repos/test_repo"
TEST_DATA_DIR = Path(__file__).parent / "data"

@fixture
def json_store():
    store_path = TEST_DATA_DIR / "test_store"
    store = JsonStore(TEST_DATA_DIR / "test_store")
    yield store

    shutil.rmtree(store_path)
    
class GitCommitContext:
    def __init__(self, repo_path, target_commit):
        self.repo = git.Repo(repo_path)
        self.target_commit = target_commit
        self.original_commit = self.repo.head.commit.hexsha

    def __enter__(self):
        self.repo.git.reset("--hard", self.target_commit)
        return self.repo

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.repo.git.reset("--hard", self.original_commit)