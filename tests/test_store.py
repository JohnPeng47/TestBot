import pytest
from pathlib import Path
from testbot.store.json_store import JsonStore
from testbot.store.models import RepoConfig, TestFileData

@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "test_store"

@pytest.fixture
def json_store(store_path):
    return JsonStore(store_path)

def test_create_repo(json_store: JsonStore):
    repo_config = RepoConfig(
        repo_name="test-repo",
        url="https://github.com/user/test-repo.git",
        source_folder="src"
    )
    
    json_store.create_repo(repo_config)
    
    # Test retrieval
    retrieved = json_store.get_repo_config("test-repo")
    assert retrieved.repo_name == repo_config.repo_name
    assert retrieved.url == repo_config.url

def test_get_nonexistent_repo(json_store):
    with pytest.raises(KeyError):
        json_store.get_repo_config("nonexistent-repo")

def test_update_testfile_data(json_store):
    test_data = TestFileData(
        id="test1",
        name="test_module.py",
        filepath="tests/test_module.py",
        targeted_files=["src/module.py"],
        test_metadata={"type": "unit"}
    )
    
    result = json_store.update_or_create_testfile_data(test_data)
    assert result.id == test_data.id
    assert result.name == test_data.name
    
    # Test retrieval
    retrieved = json_store.get_testfile_data(Path("tests/test_module.py"))
    assert retrieved.id == test_data.id
    assert retrieved.filepath == test_data.filepath

def test_get_nonexistent_testfile(json_store):
    with pytest.raises(KeyError):
        json_store.get_testfile_data(Path("nonexistent.py"))
