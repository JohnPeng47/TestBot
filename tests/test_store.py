from pathlib import Path
import pytest

from testbot.models.core import RepoConfig, TestFileData
from testbot.store.json_store import JsonStore

def test_create_repo(json_store: JsonStore):
    repo_config = RepoConfig(
        repo_name="test-repo",
        url="https://github.com/user/test-repo.git",
        source_folder="src"
    )
    
    json_store.create_repoconfig(repo_config)
    
    # Test retrieval
    retrieved = json_store.get_repo_config("test-repo")
    assert retrieved.repo_name == repo_config.repo_name
    assert retrieved.url == repo_config.url

def test_get_nonexistent_repo(json_store: JsonStore):
    with pytest.raises(KeyError):
        json_store.get_repo_config("nonexistent-repo")

def test_update_testfile_data(json_store: JsonStore):
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

def test_get_nonexistent_testfile(json_store: JsonStore):
    with pytest.raises(KeyError):
        json_store.get_testfile_data(Path("nonexistent.py"))

def test_get_testfiles_from_srcfile(json_store: JsonStore):
    # Create test data
    test_data1 = TestFileData(
        id="test1",
        name="test_module1.py", 
        filepath="tests/test_module1.py",
        targeted_files=["src/module.py"],
        test_metadata={"type": "unit"}
    )
    
    test_data2 = TestFileData(
        id="test2",
        name="test_module2.py",
        filepath="tests/test_module2.py", 
        targeted_files=["src/module.py", "src/other.py"],
        test_metadata={"type": "unit"}
    )

    # Store test data
    json_store.update_or_create_testfile_data(test_data1)
    json_store.update_or_create_testfile_data(test_data2)

    # Test retrieval
    source_file = Path("src/module.py")
    matching_tests = json_store.get_testfiles_from_srcfile(source_file)

    # Both test files should be returned since they target src/module.py
    assert len(matching_tests) == 2
    assert any(t.id == "test1" for t in matching_tests)
    assert any(t.id == "test2" for t in matching_tests)

    # Test with file that has no tests
    source_file = Path("src/no_tests.py") 
    matching_tests = json_store.get_testfiles_from_srcfile(source_file)
    assert len(matching_tests) == 0
