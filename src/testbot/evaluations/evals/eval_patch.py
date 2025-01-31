from src.testbot.store import TestBotStore, JsonStore
from src.testbot.diff import CommitDiff, DiffMode
from src.testbot.evaluations.models import EvalData, Commit
from src.testbot.evaluations.utils import get_db_session

from typing import List, Dict, Any
from sqlmodel import Session
import click

from git import Repo
from datetime import datetime
import numpy as np
from collections import defaultdict

def bucket_commits(commit_shas: list, num_buckets: int, repo_path: str = '.') -> dict:
    """
    Bucket commits based on their dates into specified number of buckets.
    
    Args:
        commit_shas: List of commit SHAs to analyze
        num_buckets: Number of buckets to segment the commits into
        repo_path: Path to the git repository (defaults to current directory)
    
    Returns:
        Dictionary with bucket ranges as keys and number of commits as values
    """
    repo = Repo(repo_path)
    
    # Get commit objects and their dates
    commit_dates = []
    for sha in commit_shas:
        try:
            commit = repo.commit(sha)
            commit_dates.append(commit.committed_datetime)
        except Exception as e:
            print(f"Error processing commit {sha}: {e}")
            continue
    
    if not commit_dates:
        return {}
    
    # Calculate bucket boundaries
    min_date = min(commit_dates)
    max_date = max(commit_dates)
    total_seconds = (max_date - min_date).total_seconds()
    bucket_size = total_seconds / num_buckets
    
    # Create buckets
    buckets = defaultdict(int)
    
    # Assign commits to buckets
    for date in commit_dates:
        bucket_index = int((date - min_date).total_seconds() // bucket_size)
        if bucket_index == num_buckets:  # Handle edge case for max date
            bucket_index -= 1
            
        # Create readable bucket range for the key
        bucket_start = min_date + np.timedelta64(int(bucket_index * bucket_size), 's')
        bucket_end = min_date + np.timedelta64(int((bucket_index + 1) * bucket_size), 's')
        bucket_key = f"{bucket_start.strftime('%Y-%m-%d %H:%M')} to {bucket_end.strftime('%Y-%m-%d %H:%M')}"
        
        buckets[bucket_key] += 1
    

    print("\nCommit Distribution across Time Buckets:")
    print("-" * 50)
    
    for bucket_range, count in buckets.items():
        print(f"Bucket {bucket_range}:")
        print(f"Number of commits: {count}")
        print(f"Visual distribution: {'#' * count}")
        print("-" * 50)


def get_commits(
    repo_name: str,
    session: Session,
    num_files: int = 1,
    num_test_files: int = 1,
    sha: str = None,
    diff_bytes: int = None
) -> List[Commit]:
    """List commits matching the given filters"""

    filters = {}
    if repo_name:
        filters["repo"] = repo_name
    if num_files is not None:
        filters["num_files"] = num_files 
    if num_test_files is not None:
        filters["num_test_files"] = num_test_files
    if sha:
        filters["sha"] = sha
    if diff_bytes is not None:
        filters["diff_bytes"] = diff_bytes

    query = session.query(Commit)
    valid_attrs = ["sha", "diff", "repo", "diff_bytes", "num_files", "num_test_files"]
    for attr, value in filters.items():
        if attr in valid_attrs:
            if attr == "diff_bytes":
                query = query.filter(getattr(Commit, attr) <= value)
            else:
                query = query.filter(getattr(Commit, attr) == value)
        else:
            raise ValueError(f"Invalid filter attribute: {attr}")
    
    commits = query.all()
    if not commits:
        print("No commits found matching filters")
        return

    return commits

@click.command()
@click.argument("repo_name", type=str)
@click.option("--num-files", default=1, help="Number of files changed")
@click.option("--num-test-files", default=1, help="Number of test files changed") 
@click.option("--sha", help="Specific commit SHA")
@click.option("--diff-bytes", type=int, help="Maximum diff size in bytes")
@click.pass_context
def eval_patch(
    repo_name: str,
    session: Session,
    num_files: int = 1,
    num_test_files: int = 1,
    sha: str = None,
    diff_bytes: int = None
):
    """Evaluates test gen on patchdiff inputs"""
    commits = get_commits(repo_name, session, num_files, num_test_files, sha, diff_bytes)
    bucket_commits(commits, 5)
    # for commit in commits:
    #     diff = CommitDiff(commit.di)

    #     new_testfunc = False
    #     new_file = False
    #     for d in diff.test_diffs():
    #         for hunk in d.hunks:
    #             if hunk.new_func:
    #                 new_testfunc = True
    #                 break
    #         if diff.creates_new_file():
    #             new_file = True
                        
    #     # only interested in tests and existing files
    #     if not new_testfunc or new_file:
    #         continue