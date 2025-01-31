from pathlib import Path
from git import Repo

from conftest import TEST_REPO, GitCommitContext, normalize_strings
from testbot.workflow.test_diff.core import TestDiffWorkflow
from testbot.llm.llm import LLMModel
from testbot.store import TestBotStore
from testbot.code import SupportedLangs
from testbot.diff import CommitDiff

import pytest

@pytest.mark.parametrize("to_merge_code,new_code,expected", [
    # Case 1: Adding a new test after existing tests
    (
        "def test_calculator():\n    calc = Calculator()\n    assert calc.add(2, 3) == 5\n",
        "def test_calculator():\n    calc = Calculator()\n    assert calc.add(2, 3) == 5\n\n# NEW CODE ALERT\ndef test_calculator_negative():\n    calc = Calculator()\n    assert calc.add(-2, -3) == -5\n",
        "def test_calculator():\n    calc = Calculator()\n    assert calc.add(2, 3) == 5\n\ndef test_calculator_negative():\n    calc = Calculator()\n    assert calc.add(-2, -3) == -5\n"
    ),
    # # Case 2: Inserting a test between existing tests
    # (
    #     "class TestUser:\n    def test_name(self):\n        assert user.name == 'John'\n\n    def test_email(self):\n        assert user.email == 'john@test.com'\n",
    #     "class TestUser:\n    def test_name(self):\n        assert user.name == 'John'\n\n# NEW CODE ALERT\n    def test_name_empty(self):\n        with pytest.raises(ValueError):\n            user.name = ''\n",  # Removed the duplicate test_email here
    #     "class TestUser:\n    def test_name(self):\n        assert user.name == 'John'\n\n    def test_name_empty(self):\n        with pytest.raises(ValueError):\n            user.name = ''\n\n    def test_email(self):\n        assert user.email == 'john@test.com'\n"
    # ),
    # Case 3: Adding multiple tests with setup
    (
        "@pytest.fixture\ndef db():\n    return MockDB()\n\ndef test_query(db):\n    result = db.query('SELECT 1')\n    assert result == [1]\n",
        "@pytest.fixture\ndef db():\n    return MockDB()\n\ndef test_query(db):\n    result = db.query('SELECT 1')\n    assert result == [1]\n\n# NEW CODE ALERT\ndef test_query_empty(db):\n    result = db.query('SELECT * FROM empty')\n    assert result == []\n\ndef test_query_error(db):\n    with pytest.raises(DBError):\n        db.query('INVALID')\n",
        "@pytest.fixture\ndef db():\n    return MockDB()\n\ndef test_query(db):\n    result = db.query('SELECT 1')\n    assert result == [1]\n\ndef test_query_empty(db):\n    result = db.query('SELECT * FROM empty')\n    assert result == []\n\ndef test_query_error(db):\n    with pytest.raises(DBError):\n        db.query('INVALID')\n"
    )
])
def test_merge_code(to_merge_code, new_code, expected):
    lm = LLMModel(use_cache=True)
    workflow = TestDiffWorkflow(
        commit=None,
        repo_name="JohnPeng47/TestRepo", 
        repo_path=TEST_REPO,
        lm=lm,
        store=None,
        lang=SupportedLangs.Python
    )
    merged_code = workflow._merge_code(new_code, to_merge_code)
    assert normalize_strings(merged_code) == normalize_strings(expected)


def test_diff_workflow(json_store: TestBotStore):
    with GitCommitContext(TEST_REPO, "dcc6c2ca16be8065b1a1705c1b60e01cf539195d"):
        # Create a simple test file change
        test_file = TEST_REPO / "important_file.py"
        test_file.write_text("def add(a, b): return a + b")
        
        # Stage the change
        repo = Repo(TEST_REPO)
        repo.index.add(["important_file.py"])

        # Create commit diff
        patch = repo.git.diff('HEAD', cached=True)
        commit_diff = CommitDiff(patch)
        
        # Initialize and run workflow
        
        # !!!! CACHE DOESNT WORK !!!
        lm = LLMModel(use_cache=True)
        workflow = TestDiffWorkflow(
            commit=commit_diff,
            repo_name="JohnPeng47/TestRepo", 
            repo_path=TEST_REPO,
            lm=lm,
            store=json_store,
            lang=SupportedLangs.Python
        )
        
        workflow.run()

        # Cleanup
        if test_file.exists():
            test_file.unlink()

