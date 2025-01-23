from ..base import WorkFlow
from testbot.code import SupportedLangs
from testbot.llm import LLMModel
from testbot.diff import CommitDiff
from testbot.store.store import TestBotStore

from pathlib import Path

class TestDiffWorkflow(WorkFlow):
    """Generates test cases for a code diff"""
    def __init__(self, 
                 commit: CommitDiff,
                 repo_name: str, 
                 repo_path: Path, 
                 lm: LLMModel,
                 store: TestBotStore,
                 lang: SupportedLangs):
        super().__init__(repo_name, repo_path, lm, store, lang)
        
        self._commit = commit
    
    