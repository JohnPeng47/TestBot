from pathlib import Path

# TODO: replace this with platformdir
TESTBOT_DATA_PATH = Path(__file__).parent.parent.parent / ".testbot"

STORE_PATH = TESTBOT_DATA_PATH / "store"
LOG_DIR = TESTBOT_DATA_PATH / "logs"

BRAINTRUST_PROJ_NAME = "testbot"