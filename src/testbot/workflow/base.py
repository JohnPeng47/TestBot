from testbot.code import SupportedLangs, Linter
from testbot.llm import LLMModel
from testbot.store.store import TestBotStore

from pathlib import Path

class WorkFlow:
    def __init__(self, 
                 repo_name: str, 
                 repo_path: Path, 
                 lm: LLMModel,
                 store: TestBotStore,
                 lang: SupportedLangs):
        self._lang = lang
        self._linter = Linter(lang)
        self._repo_name = repo_name
        self._repo_path = repo_path

        self._lm = lm
        self._store = store