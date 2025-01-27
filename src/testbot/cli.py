import click
from pathlib import Path
from git import Repo
from datetime import datetime
import sys

from testbot.store import JsonStore
from testbot.workflow import InitRepo, TestDiffWorkflow
from testbot.llm import LLMModel
from testbot.utils import load_env
from testbot.diff import CommitDiff

TEST_PATCH = """
diff --git a/a.py b/a.py
index 0d1bea4..c99da5d 100644
--- a/a.py
+++ b/a.py
@@ -10,4 +10,8 @@ def divide(a, b):
    \"\"\"Divide a by b.\"\"\"
    if b == 0:
        raise ValueError(\"Cannot divide by zero\")

+def average(**args):
+    total = sum(args)
+    return total / len(args)
"""

def install_hooks(repo_path=".", dry_run=False):
    """Install git hooks for the repository"""
    try:
        repo = Repo(repo_path)
        hook_path = Path(repo.git_dir) / "hooks" / "pre-commit"
        
        # Create hook script
        hook_content = f"""#!/bin/bash
python {Path(__file__).resolve()} repo pre-commit {f"--dry-run" if dry_run else ""}
"""
        
        # Write and make executable
        hook_path.write_text(hook_content)
        hook_path.chmod(0o755)
        print(f"Installed pre-commit hook to {hook_path}")
        return 0
        
    except Exception as e:
        print(f"Error installing hooks: {e}")
        return 1

def delete_hooks(repo_path="."):
    """Delete git hooks for the repository"""
    try:
        repo = Repo(repo_path)
        hook_path = Path(repo.git_dir) / "hooks" / "pre-commit"
        
        if hook_path.exists():
            hook_path.unlink()
            print(f"Deleted pre-commit hook at {hook_path}")
            return 0
        else:
            print("No pre-commit hook found")
            return 0
            
    except Exception as e:
        print(f"Error deleting hooks: {e}")
        return 1

@click.group()
def cli():
    """TestBot CLI tool for managing test repositories"""
    pass

@cli.group()
def repo():
    """Commands for managing test repositories"""
    pass

# DESIGN: consider adding an interactive test environment to this
@repo.command()
@click.option("--dry-run", is_flag=True, help="For testing if hook installed")
def pre_commit(dry_run):
    """Run pre-commit checks on staged changes"""

    if dry_run:
        # NOTE: writing to stderr cuz git hooks convention, prolly unbuffered, 
        # and clears room for scripts that are depending on stdout output to be piped
        sys.stderr.write("Install success!\n")
        return "Install success!"

    store = JsonStore()
    try:
        repo = Repo('.')
        patch = repo.git.diff('HEAD', cached=True)  # 'cached' means staged changes
        if not patch:
            sys.stderr.write("No staged changes found\n")
            sys.exit(0)
        
        commit = CommitDiff(
            patch=patch,
            timestamp=datetime.now().isoformat()
        )
        sys.stderr.write(f"Changed files: {commit.src_files}\n")

        workflow = TestDiffWorkflow(commit, LLMModel(), store)
        workflow.run()
    except Exception as e:
        sys.stderr.write(f"Error in pre-commit check: {e}\n")
        sys.exit(1)


@repo.command()
@click.option("--dry-run", is_flag=True, help="For testing if hook installed")
def pre_commit(dry_run):
    """Run pre-commit checks on staged changes"""

    if dry_run:
        # NOTE: writing to stderr cuz git hooks convention, prolly unbuffered, 
        # and clears room for scripts that are depending on stdout output to be piped
        sys.stderr.write("Install success!\n")
        return "Install success!"

    store = JsonStore()
    try:
        repo = Repo('.')
        patch = repo.git.diff('HEAD', cached=True)  # 'cached' means staged changes
        if not patch:
            sys.stderr.write("No staged changes found\n")
            sys.exit(0)
        
        commit = CommitDiff(
            patch=patch,
            timestamp=datetime.now().isoformat()
        )
        sys.stderr.write(f"Changed files: {commit.src_files}\n")

        workflow = TestDiffWorkflow(commit, LLMModel(), store)
        workflow.run()
    except Exception as e:
        sys.stderr.write(f"Error in pre-commit check: {e}\n")
        sys.exit(1)

@repo.command()
def test_pre_commit():
    """Run pre-commit checks on staged changes"""
    store = JsonStore()
    try:
        commit = CommitDiff(
            patch=TEST_PATCH,
            timestamp=datetime.now().isoformat()
        )
        sys.stderr.write(f"Changed files: {commit.src_files}\n")
        workflow = TestDiffWorkflow(commit, 
                                    "test_repo", 
                                    Path("tests/test_repos/test_repo"),
                                    LLMModel(), 
                                    store,
                                    "python")
        workflow.run()
    except Exception as e:
        import traceback
        sys.stderr.write(f"Error in pre-commit check: {e}\n")
        sys.stderr.write(f"Stacktrace:\n{traceback.format_exc()}\n")
        sys.exit(1)

@repo.command()
@click.argument("repo_path")
@click.option("--language", default=None)
@click.option("--limit", type=int, default=None, help="Limit the number of test files to map back to source")
def init(repo_path, language, limit):
    """Initialize a new test repository"""
    store = JsonStore()
    repo_path = Path(repo_path)
    repo = store.get_repoconfig(
        lambda x: x.source_folder == str(repo_path.resolve())
    )
    if repo:
        store.delete_repoconfig(repo.repo_name)

    workflow = InitRepo(Path(repo_path), LLMModel(), store, language=language, limit=limit)
    workflow.run()
    install_hooks(repo_path=repo_path)

# @repo.command()
# def delete():
#     """Delete an existing test repository"""
#     store = JsonStore()

def main():
    load_env() # load LLM API keys
    
    cli()

if __name__ == "__main__":
    main()