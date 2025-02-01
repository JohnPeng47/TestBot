from typing import Dict, Callable, List, Any
from dataclasses import dataclass
from sqlmodel import SQLModel, Field, JSON, Relationship
from pydantic import BaseModel, Field as pyField
from abc import ABC

# NOTE: it would probably be simpler if we just combined Commit and PatchEvalConfig 
# into one but separating them gives us some flexibility to select which commits
# to commit to evaluation data
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

# Braintrust stuff
class DatasetInput(BaseModel):
    prompt_args: Dict
    run_args: Dict = pyField(default_factory=dict)
    
class BraintrustDataset(BaseModel):
    input: DatasetInput
    expected: Dict

class ToDataset(ABC):
    def to_dataset(self) -> BraintrustDataset:
        raise NotImplementedError

@dataclass
class EvalData:
    name: str
    dataset: Dict
    eval_fn: Callable
    score_fns: List[Callable]