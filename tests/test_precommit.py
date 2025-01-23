from pathlib import Path
from git import Repo
import os
import subprocess
from contextlib import chdir

from conftest import TEST_REPO, GitCommitContext
from testbot.cli import install_hooks, delete_hooks

def test_hook_install():
    with GitCommitContext(TEST_REPO, "dcc6c2ca16be8065b1a1705c1b60e01cf539195d"):
        # Verify no hook exists
        repo = Repo(TEST_REPO)
        hook_path = Path(repo.git_dir) / "hooks" / "pre-commit"
        assert not hook_path.exists()

        # Install hook
        result = install_hooks(str(TEST_REPO))
        assert result == 0

        # Create and stage a test file
        test_file = TEST_REPO / "test.txt"
        test_file.write_text("test content")
        repo.index.add(["test.txt"])
        
        # Try to commit - should trigger pre-commit hook
        try:
            repo.index.commit("test commit")
        finally:
            # Cleanup
            if test_file.exists():
                os.remove(test_file)
            result = delete_hooks(str(TEST_REPO))
            assert result == 0

def test_precommit_hook():
    with GitCommitContext(TEST_REPO, "dcc6c2ca16be8065b1a1705c1b60e01cf539195d"):
        # Read existing files
        a_file = TEST_REPO / "a.py"
        b_file = TEST_REPO / "b.py"
        
        a_content = a_file.read_text()
        b_content = b_file.read_text()

        # Append new content
        a_append = """
def square_and_normalize(a, b):
    squared = a * a
    if b == 0:
        raise ValueError("Cannot normalize by zero")
    return squared / b

def calculate_ratio(a, b):
    if b == 0:
        raise ValueError("Cannot calculate ratio with zero denominator") 
    squared_a = a * a
    squared_b = b * b
    return squared_a / b
"""
        test_b_append = """
def find_median(items):
    if not items:
        raise ValueError("Cannot find median of empty list")
    sorted_items = sorted(items)
    length = len(sorted_items)
    mid = length // 2
    if length % 2 == 0:
        return (sorted_items[mid - 1] + sorted_items[mid]) / 2
    return sorted_items[mid]

def get_unique_count(items):
    return len(set(items))
"""
        # Write updated content
        a_file.write_text(a_content + a_append)
        b_file.write_text(b_content + test_b_append)

        # Stage the changes
        with chdir(TEST_REPO):
            repo = Repo(".")
            repo.index.add(["a.py", "b.py"])

            # Try to commit and check exit code
            result = subprocess.run(["git", "commit", "-m", "test commit"], 
                                cwd=str(TEST_REPO),
                                capture_output=True)
            
            assert "Changed files: ['a.py', 'b.py']" in result.stderr.decode("utf-8")
            assert result.returncode == 0

