from enum import Enum

class SupportedLangs(str, Enum):
    Python = "python"

EXTENSIONS = {
    SupportedLangs.Python: ".py"
}

TEST_PATTERNS = {
    SupportedLangs.Python: ["test_*.py", "*_test.py"]
}