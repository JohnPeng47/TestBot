from git import Repo
from pathlib import Path
import subprocess
import os

TEST_REPO = Path("tests/test_repos/test_repo")

repo = Repo(TEST_REPO)
original_dir = os.getcwd()

try:
    os.chdir(repo.working_dir)
    repo.index.add(["."])
    # Any subprocess calls here will also use the correct working directory
    result = subprocess.run(["git", "commit", "-m", "test commit"], 
                            capture_output=True)
    print(result.stdout)
finally:
    os.chdir(original_dir)