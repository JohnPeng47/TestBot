import click
from pathlib import Path
from git import Repo
from datetime import datetime
import sys
import os

from testbot.terminal import IO
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

@click.group()
@click.option("--debug", is_flag=True, help="Enable debug mode")
def cli(debug):
    """TestBot CLI tool for managing test repositories"""
    load_env()
    print(debug)

    import sys
    sys.exit(0)


@cli.command()
def staged():
    """Generate tests for staged changes"""
    repo_path = Path(os.getcwd()).resolve() # git sets CWD when calling pre-commit

    io = IO()
    if not io.input("Proceed with TestBot test generation [y/N]: ", 
               validator=lambda x: x.lower() == "y"):
        print("Proceeding with normal git commit process\n")
        sys.exit(0)

    store = JsonStore()
    try:
        repo = Repo(repo_path)
        patch = repo.git.diff('HEAD', cached=True)  # 'cached' means staged changes
        if not patch:
            print("No staged changes found\n")
            sys.exit(0)
        
        commit = CommitDiff(
            patch=patch,
            timestamp=datetime.now().isoformat()
        )
        print(f"Changed files: {commit.src_files}\n")

        workflow = TestDiffWorkflow(commit, repo_path, LLMModel(), store, io)
        workflow.run()

        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"Error in pre-commit check: {e}\n")
        print(f"Stacktrace:\n{traceback.format_exc()}\n")
        sys.exit(1)

@cli.command()
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

# @repo.command()
# def delete():
#     """Delete an existing test repository"""
#     store = JsonStore()

def main():    
    cli()

if __name__ == "__main__":
    main()