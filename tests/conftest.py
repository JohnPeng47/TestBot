from pytest import fixture
import git

from pathlib import Path
import shutil
from testbot.store.json_store import JsonStore
from testbot.utils import GitCommitContext

TEST_REPO = Path(__file__).parent / "test_repos/test_repo"
TEST_DATA_DIR = Path(__file__).parent / "data"

@fixture
def json_store():
    store_path = TEST_DATA_DIR / "test_store"
    store = JsonStore(TEST_DATA_DIR / "test_store")
    yield store

    shutil.rmtree(store_path)

def normalize_strings(string):
    return string.strip().replace("\r\n", "\n")