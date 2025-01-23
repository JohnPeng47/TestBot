import click
from pathlib import Path
from git import Repo
from datetime import datetime

from testbot.store import JsonStore
from testbot.workflow import InitRepo, TestDiffWorkflow
from testbot.llm import LLMModel
from testbot.utils import load_env
from testbot.diff import CommitDiff

def install_hooks(repo_path=".", dry_run = True):
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

# TOOD: let's also make this work on staging as wlell
@repo.command()
@click.option("--dry-run", is_flag=True, help="For testing if hook installed")
def pre_commit(dry_run):
    """Run pre-commit checks on staged changes"""

    if dry_run:
        return "Install success!"

    store = JsonStore()
    try:
        repo = Repo('.')
        diff = repo.index.diff('HEAD', create_patch=True)
        if not diff:
            print("No staged changes found")
            return 0
            
        patch = '\n'.join(d.diff.decode('utf-8') for d in diff)
        
        # Create commit diff object
        commit = CommitDiff(
            patch=patch,
            timestamp=datetime.now().isoformat()
        )
        
        workflow = TestDiffWorkflow(commit, LLMModel(), store)
        return workflow.run()
        
    except Exception as e:
        print(f"Error in pre-commit check: {e}")
        return 1

@repo.command()
@click.argument("repo_path")
@click.option("--language", default=None)
@click.option("--limit", type=int, default=None, help="Limit the number of test files to map back to source")
def init(repo_path, language, limit):
    """Initialize a new test repository"""
    store = JsonStore()
    workflow = InitRepo(Path(repo_path), LLMModel(), store, language=language, limit=limit)
    workflow.run()
    # install_hooks()

# @repo.command()
# def delete():
#     """Delete an existing test repository"""
#     store = JsonStore()
#     # TODO: Implement repository deletion


def main():
    load_env() # load LLM API keys

    cli()

if __name__ == "__main__":
    main()