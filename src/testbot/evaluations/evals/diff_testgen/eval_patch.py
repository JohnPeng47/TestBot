from src.testbot.diff import CommitDiff
from src.testbot.evaluations.models import Commit
from src.testbot.utils import GitCommitContext, is_later_commit
from src.testbot.evaluations.utils import get_db_session_and_store
from src.testbot.evaluations.models import BraintrustDataset, ToDataset, DatasetInput
from src.testbot.workflow.init.lmp import IdentifyModules 
from src.testbot.llm.llm import LLMModel
from src.testbot.workflow import InitRepo

from sqlmodel import Field, Relationship, SQLModel, Session, JSON, PrimaryKeyConstraint
from pathlib import Path
from typing import List
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

@eval_patch.command(name="inspect")
@click.argument("repo_name", type=str)
@click.option("--num-files", default=1, help="Number of files changed")
@click.option("--num-test-files", default=1, help="Number of test files changed") 
@click.option("--sha", help="Specific commit SHA")
@click.option("--diff-bytes", type=int, help="Maximum diff size in bytes")
@click.pass_context
def inspect_commit_data(
    ctx: click.Context,
    repo_name: str,
    num_files: int = 1,
    num_test_files: int = 1,
    sha: str = None,
    diff_bytes: int = None
):
    """Use this function to inspect commit data"""    
    session, store = get_db_session_and_store(ctx)
    commits = get_commits(repo_name, session, num_files, num_test_files, sha, diff_bytes)
    repo_config = store.get_repoconfig(lambda x: x.repo_name == repo_name)
    repo_path = Path(repo_config.source_folder)
    filtered_commits = []
    
    for commit in commits:
        diff = CommitDiff(commit.diff)
        after_commit = is_later_commit(commit.sha, sha, repo_path)
        new_testfunc = diff.contains_newtest()
        new_file = False
        testfile = ""

        for d in diff.test_diffs():
            if d.creates_new_file():
                new_file = True
                break

        # only interested in tests and existing files
        if not new_testfunc or new_file or not after_commit:
            continue

        filtered_commits.append(commit)

    for i, c in enumerate(filtered_commits, start=1):
        print(f"{i}||________________________________________________________________")
        print("SHA: ", c.sha)
        print(c.diff)


class DiffTestgenDataset(SQLModel, ToDataset, table=True):
    __table_args__ = (
        PrimaryKeyConstraint('sha', 'dataset_name'),
        {'extend_existing': True}  # Dictionary must be last
    )

    sha: str = Field(foreign_key="commit.sha")
    patch: str
    dataset_name: str
    source_files: List[str] = Field(default={}, sa_type=JSON)
    commit: Commit = Relationship() 

    def to_dataset(self) -> BraintrustDataset:
        return BraintrustDataset(
            input=DatasetInput(
                prompt_args={"patch": self.patch, "source_files": self.source_files},
                run_args={}
            ),
            expected={"patch": self.patch}
        )

# TMRW:
# - think through decision to use latest commit and go backwards with optional stopping commit,
# instead of going forward from a starting commit; reason being that files are more likely to be missing
# (going forward) than to be deleted/renamed (going backwards) 
@eval_patch.command(name="create-dataset")
@click.argument("repo_name", type=str)
@click.option("--name", default=Path(__file__).stem + "_default", help="Name of the dataset")
@click.option("--num-files", default=1, help="Number of files changed")
@click.option("--num-test-files", default=1, help="Number of test files changed") 
@click.option("--sha", help="Specific commit SHA")
@click.option("--diff-bytes", type=int, help="Maximum diff size in bytes")
@click.option("--commits-file", type=click.Path(exists=True), help="Path to file containing commit SHAs")
@click.pass_context
def construct_dataset(
    ctx: click.Context,
    repo_name: str,
    num_files: int = 1,
    num_test_files: int = 1,
    name: str = "",
    sha: str = None,
    diff_bytes: int = None,
    commits_file: str = None
):
    session, store = get_db_session_and_store(ctx)
    lm = LLMModel()
    repo_config = store.get_repoconfig(lambda x: x.repo_name == repo_name)
    if not repo_config:
        raise Exception(f"Repository {repo_name} not found in database")
    
    repo_path = Path(repo_config.source_folder)
    if commits_file:
        with open(commits_file) as f:
            commit_shas = [line.strip() for line in f.readlines()]
        commits = session.query(Commit).filter(Commit.sha.in_(commit_shas)).all()
    else:
        commits = get_commits(repo_name, session, num_files, num_test_files, sha, diff_bytes)

    for commit in commits:
        commit_diff = CommitDiff(commit.diff)
        for test_fp in commit_diff.test_files:
            print("[BUILD-DATASET] Processing test file: ", test_fp)
            print("[PATCH]:\n", commit_diff)

            test_content = open(repo_path / test_fp, "r").read()
            target_files = IdentifyModules().invoke(
                lm,
                model_name = "claude",
                test_file = test_content,
                repo_path = repo_path
            )

            print("[TARGET-FILES]: ", target_files)
            dataset = DiffTestgenDataset(
                dataset_name=name,
                sha=commit.sha,
                patch=commit.diff,
                source_files=[str(fp) for fp in target_files],
                commit=commit
            )
            session.merge(dataset)
            session.commit()

        return
    
