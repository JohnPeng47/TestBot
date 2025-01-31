from testbot.code import SupportedLangs
from testbot.llm.llm import LLMModel
from testbot.store.store import TestBotStore

from pathlib import Path

class WorkFlow:
    """
    Base class for workflow, handles generic logging
    """
    def __init__(self, 
                 lm: LLMModel,
                 store: TestBotStore):
        self._lm = lm
        self._store = store

    def run(self):
        raise NotImplementedError("Run method not implemented")