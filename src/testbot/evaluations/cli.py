import click
from braintrust import Eval
from typing import NewType
import git
from pathlib import Path
import traceback
from sqlmodel import Session, create_engine, SQLModel
from functools import wraps

from testbot.diff import CommitDiff
from testbot.llm.llm import LLMModel, num_tokens_from_string
from testbot.config import BRAINTRUST_PROJ_NAME
from testbot.utils import load_env

from .models import Commit
from .utils import get_db_session
from .evals import eval_patch

## import evals
from testbot.workflow.test_diff.evals import FILTER_SRC_FILES_CONSISTENCY

SHA = NewType("SHA", str)

from logging import getLogger

logger = getLogger("get_pr")

# TODO: convert this into a eval subcommand
ALL_EVALS = [
    FILTER_SRC_FILES_CONSISTENCY
]

@click.group()
@click.option("--db-path", default="evaluations.db", help="Path to SQLite database")
@click.pass_context
def eval_cli(ctx: click.Context, db_path: str):
    """Run evaluations"""
    ctx.ensure_object(dict)
    
    sqlite_url = f"sqlite:///{db_path}"
    engine = create_engine(sqlite_url, echo=False)
    
    # Create tables only once at startup
    SQLModel.metadata.create_all(engine)
    
    ctx.obj["engine"] = engine

# add evaluations as subcommands    
eval_cli.add_command(eval_patch)

@eval_cli.command()
@click.argument("repo_path", type=str)
@click.pass_context
def build_commits(ctx: click.Context, repo_path: str):
    """Download commits for a repository"""
    with get_db_session(ctx) as session:
        repo_path = Path(repo_path)
        repo = git.Repo(repo_path)
        commits = list(repo.iter_commits())

        print(f"Downloading {len(commits)} commits")

        for i, commit in enumerate(commits):
            try:
                print(f"{i}/{len(commits)}")
                
                num_files = 0
                num_test_files = 0
                diff_bytes = 0
                
                if not commit.parents:
                    print("Skipping commit without parents")
                    continue

                git_diff = repo.git.diff(commit.parents[0], commit, unified=3)
                diff = CommitDiff(git_diff)

                is_test_modification = False
                num_files += len(diff.code_files)
                num_test_files += len(diff.test_files)
                code_diff_text = "".join(str(d) for d in diff.code_diffs())
                test_diff_text = "".join(str(d) for d in diff.test_diffs())
                
                combined_diff_text = code_diff_text + test_diff_text
                diff_bytes += num_tokens_from_string(combined_diff_text)

                if num_files > 0 and num_test_files > 0:
                    print(f"Commit {commit} has {num_files} files and {num_test_files} test files")
                    if num_files == 1 and num_test_files == 1:
                        print(diff)

                    commit_obj = Commit(
                        sha=commit.hexsha,
                        diff=git_diff,
                        repo=repo_path.name,
                        num_files=num_files,
                        num_test_files=num_test_files,
                        diff_bytes=diff_bytes,
                        timestamp=diff.timestamp,
                        merge_commit=False,
                        merge_parent=None,
                        is_test_modification=is_test_modification
                    )
                    
                    existing = session.get(Commit, commit.hexsha)
                    if existing:
                        for key, value in commit_obj.dict().items():
                            setattr(existing, key, value)
                    else:
                        session.add(commit_obj)
                    session.commit()

            except Exception as e:
                print(traceback.format_exc())
                print(f"Downloading {commit} failed")
                continue

@eval_cli.command()
@click.option("--repo", help="Filter by repository name")
@click.option("--num-files", type=int, default=1, help="Filter by number of files")
@click.option("--num-test-files", type=int, default=1, help="Filter by number of test files")
@click.option("--sha", help="Filter by commit SHA")
@click.option("--diff-bytes", type=int, help="Filter by diff size in bytes")
@click.pass_context
def list_commits(ctx, repo, num_files, num_test_files, sha, diff_bytes):
    """List commits matching the given filters"""
    with get_db_session(ctx) as session:
        filters = {}
        if repo:
            filters["repo"] = repo
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
            click.echo("No commits found matching filters")
            return

        MAX_BYTES = 1000
        for commit in commits:
            # if sum(len(d.body) for d in diff.code_diffs()) > MAX_BYTES:
            #     print(f)
            #     continue

            # and includes new test definition
            # we can look for

            # diff = CommitDiff(commit.diff)
            # new_testfunc = False
            # for d in diff.test_diffs():
            #     for hunk in d.hunks:
            #         if hunk.new_func:
            #             new_testfunc = True
            #             break
                     
            # if not new_testfunc:
            #     continue
        

            click.echo(f"-------------------------Diff LEN({len(str(commit))})-----------------------------")
            click.echo(commit.diff)
            # click.echo(f"SHA: {commit.sha}")
            # click.echo(f"Repo: {commit.repo}")
            # click.echo(f"Files: {commit.num_files}")
            # click.echo(f"Test Files: {commit.num_test_files}")
            click.echo("-----------------------------------------------------------------------------------")

@eval_cli.command()
@click.argument("eval_name")
@click.option("--model-name", default="claude", help="Model identifier")
@click.option("--iters", default=15, help="Number of iterations to run", type=int)
@click.option("--experiment-suffix", "-s", help="Suffix for experiment name")
@click.option("--comments", "-m", help="Comments for the evaluation")
def run(eval_name: str,
        model_name: str,
        iters: int,
        experiment_suffix: str,
        comments: str):
    """Run a specific evaluation"""
    load_env()
    model = LLMModel()

    for eval_data in ALL_EVALS:
        if eval_data.name == eval_name:
            # add run args to each datum for the eval
            for datum in eval_data.dataset:
                datum["input"]["run_args"] = {
                    "model": model,
                    "model_name": model_name,
                    "iters": iters
                }

            experiment_name = eval_data.name + "_" + model_name + "_" + str(iters)
            if experiment_suffix:
                experiment_name += "-" + experiment_suffix

            metadata = {
                "comments": comments or ""
            }

            Eval(
                BRAINTRUST_PROJ_NAME,
                eval_data.dataset,
                eval_data.eval_fn,
                eval_data.score_fns,
                experiment_name=experiment_name,
                metadata=metadata
            )


if __name__ == "__main__":
    
    eval_cli()