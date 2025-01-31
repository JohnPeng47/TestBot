from src.testbot.store import TestBotStore, JsonStore
from src.testbot.diff import CommitDiff, DiffMode
from src.testbot.evaluations.models import EvalData, Commit, RepoEvalConfig
from src.testbot.utils import GitCommitContext
from src.testbot.evaluations.utils import get_db_session_and_store

from src.testbot.llm.llm import LLMModel
from src.testbot.workflow import InitRepo

from pathlib import Path
from typing import List, Dict, Any
from sqlmodel import Session
import click

from git import Repo
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
    commit_info = []  # Store tuples of (date, sha)
    for sha in commit_shas:
        try:
            commit = repo.commit(sha)
            commit_dates.append(commit.committed_datetime)
            commit_info.append((commit.committed_datetime, sha))
        except Exception as e:
            # print("ERROR!")
            print(f"Error processing commit {sha}: {e}")
            continue
    
    if not commit_dates:
        return {}
    
    # Calculate bucket boundaries
    min_date = min(commit_dates)
    max_date = max(commit_dates)
    total_seconds = (max_date - min_date).total_seconds()
    bucket_size = total_seconds / num_buckets
    
    # Create buckets with SHA tracking
    buckets = defaultdict(lambda: {"count": 0, "shas": []})
    
    # Assign commits to buckets
    for date, sha in commit_info:
        bucket_index = int((date - min_date).total_seconds() // bucket_size)
        if bucket_index == num_buckets:  # Handle edge case for max date
            bucket_index -= 1
            
        # Create readable bucket range for the key
        bucket_start = min_date + np.timedelta64(int(bucket_index * bucket_size), 's')
        bucket_end = min_date + np.timedelta64(int((bucket_index + 1) * bucket_size), 's')
        bucket_key = f"{bucket_start.strftime('%Y-%m-%d %H:%M')} to {bucket_end.strftime('%Y-%m-%d %H:%M')}"
        
        buckets[bucket_key]["count"] += 1
        buckets[bucket_key]["shas"].append((date, sha))

    print("\nCommit Distribution across Time Buckets:")
    print("-" * 50)
    
    for bucket_range, data in buckets.items():
        count = data["count"]
        shas = sorted(data["shas"])  # Sort by date
        first_sha = shas[0][1][:8]  # Get first 8 chars of SHA
        last_sha = shas[-1][1][:8]  # Get first 8 chars of SHA
        
        print(f"Bucket {bucket_range}:")
        print(f"Number of commits: {count}")
        print(f"First commit SHA: {first_sha}")
        print(f"Last commit SHA: {last_sha}")
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

@click.group()
def eval_patch():
    """Evaluate test generation on patch diffs"""
    pass

@eval_patch.command(name="create")
@click.argument("repo_path", type=str)
@click.argument("sha", type=str)
@click.pass_context
def add_repo(
    ctx: click.Context, 
    repo_path: str, 
    sha: str
):
    """Creates a RepoConfig at a specific commit SHA"""
    with GitCommitContext(repo_path, sha) as commit_ctx:
        repo_path = Path(repo_path)
        _, store = get_db_session_and_store(ctx)
        model = LLMModel()
        repo = store.get_repoconfig(
            lambda x: x.source_folder == str(repo_path.resolve())
        )
        if repo:
            raise Exception(f"Repository {repo.repo_name} already exists!")

        workflow = InitRepo(Path(repo_path), model, store)
        workflow.run()
        
        print("Initialization cost: ", model.get_cost())


@eval_patch.command(name="run")
@click.argument("repo_name", type=str)
@click.argument("repo_path", type=str)
@click.option("--num-files", default=1, help="Number of files changed")
@click.option("--num-test-files", default=1, help="Number of test files changed") 
@click.option("--sha", help="Specific commit SHA")
@click.option("--diff-bytes", type=int, help="Maximum diff size in bytes")
@click.pass_context
def run_eval(
    ctx: click.Context,
    repo_name: str,
    repo_path: str,
    num_files: int = 1,
    num_test_files: int = 1,
    sha: str = None,
    diff_bytes: int = None
):
    """Evaluates test gen on patchdiff inputs"""
    store = JsonStore()
    
    with get_db_session_and_store(ctx) as (session, store):
        commits = get_commits(repo_name, session, num_files, num_test_files, sha, diff_bytes)

        # bucket_commits([c.sha for c in commits], 5, repo_path=repo_path)
        filtered_commits = []
        
        for commit in commits:
            diff = CommitDiff(commit.di)
            new_testfunc = False
            new_file = False
            for d in diff.test_diffs():
                for hunk in d.hunks:
                    if hunk.new_func:
                        new_testfunc = True
                        break
                if diff.creates_new_file():
                    new_file = True
                            
            # only interested in tests and existing files
            if not new_testfunc or new_file:
                continue