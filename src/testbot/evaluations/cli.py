import click
from braintrust import Eval
from typing import NewType
import git
from pathlib import Path
import traceback
from sqlmodel import Session, create_engine, SQLModel
from functools import wraps

from testbot.store import JsonStore
from testbot.diff import CommitDiff
from testbot.llm.llm import LLMModel, num_tokens_from_string
from testbot.config import BRAINTRUST_PROJ_NAME
from testbot.utils import load_env

from .models import Commit
from .utils import get_db_session_and_store
from .evals import eval_patch, DiffTestgenDataset


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
    SQLModel.metadata.create_all(engine, checkfirst=True)

    store = JsonStore()
    
    ctx.obj["engine"] = engine
    ctx.obj["store"] = store

# add evaluations as subcommands    
eval_cli.add_command(eval_patch)

@eval_cli.command()
@click.option("--repo", help="Filter by repository name")
@click.option("--num-files", type=int, default=1, help="Filter by number of files")
@click.option("--num-test-files", type=int, default=1, help="Filter by number of test files")
@click.option("--sha", help="Filter by commit SHA")
@click.option("--diff-bytes", type=int, help="Filter by diff size in bytes")
@click.pass_context
def list_commits(ctx, repo, num_files, num_test_files, sha, diff_bytes):
    """List commits matching the given filters"""
    with get_db_session_and_store(ctx) as (session, _):
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