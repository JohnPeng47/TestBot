from enum import Enum

class SupportedLangs(str, Enum):
    Python = "python"

class Linter:
    def __init__(self, lang: SupportedLangs):
        self._lang = lang
