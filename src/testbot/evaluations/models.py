from typing import Dict, Callable, List
from dataclasses import dataclass
from sqlmodel import SQLModel, Field

class Commit(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}  # Add this line

    sha: str = Field(primary_key=True)
    diff: str
    repo: str
    diff_bytes: int = None
    num_files: int = None
    num_test_files: int = None

    def to_commit(self):
        return Commit(
            sha=self.sha,
            diff=self.diff,
            repo=self.repo
        )

class RepoEvalConfig(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}  # Add this line

    repo_name: str
    sha: str = Field(primary_key=True)

@dataclass
class EvalData:
    name: str
    dataset: Dict
    eval_fn: Callable
    score_fns: List[Callable]