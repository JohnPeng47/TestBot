from typing import NewType
import git

from testbot.diff import CommitDiff
from testbot.llm.llm import num_tokens_from_string

SHA = NewType("SHA", str)

from logging import getLogger
from pathlib import Path

logger = getLogger("get_pr")


def download_commits(repo_path: Path):
    repo = git.Repo(repo_path)
    commits = list(repo.iter_commits())

    print(f"Downloading {len(commits)} commits")

    # Figure out why this keeps failing
    for commit in commits:
        try:
            num_files = 0
            num_test_files = 0
            diff_bytes = 0

            git_diff = repo.git.diff(commit.parents[0], commit, unified=3)
            diff = CommitDiff(git_diff)

            is_test_modification = False
            num_files += len(diff.code_files)
            num_test_files += len(diff.test_files)
            code_diff_text = "".join(str(d) for d in diff.code_diffs())
            test_diff_text = "".join(str(d) for d in diff.test_diffs())
            combined_diff_text = code_diff_text + test_diff_text
            diff_bytes += num_tokens_from_string(combined_diff_text)

            # want to find if this is new test or modifying existing
            # if not is_test_modification:
            #     for hunk in diff.hunks:
            #         func_declare = hunk.new_func
            #         if func_declare:
            #             is_test_modification = True
            #             break

            if num_files > 0 and num_test_files > 0:
                print(f"Commit {commit} has {num_files} files and {num_test_files} test files")
                if num_files == 1 and num_test_files == 1:
                    print(diff)
                    
            # commit = Commit(
            #     sha=commit.hexsha,
            #     diff=git_diff,
            #     repo=repo_path.name,
            #     num_files=num_files,
            #     num_test_files=num_test_files,
            #     diff_bytes=diff_bytes,
            #     # take the timestamp of the last commit
            #     timestamp=diff.timestamp,
            #     merge_commit=False,
            #     merge_parent=None,
            #     is_test_modification=is_test_modification,
            # )

            # commit_db.upsert(commit)
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            print(f"Downloading {commit} failed")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download commits for a repository.")
    parser.add_argument("repo_path", type=str, help="Repository name")
    args = parser.parse_args()

    repo_path = Path(args.repo_path)

    download_commits(repo_path)