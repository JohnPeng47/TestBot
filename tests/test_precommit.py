from pathlib import Path
from git import Repo
import os

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