from testbot.workflow.init.core import InitRepo
from testbot.models import RepoConfig
from testbot.llm.llm import LLMModel
from testbot.store import TestBotStore

from conftest import TEST_REPO, GitCommitContext

def test_init_repo_creates_config(json_store: TestBotStore, tmp_path):
    with GitCommitContext(TEST_REPO, "dcc6c2ca16be8065b1a1705c1b60e01cf539195d"):
        lm = LLMModel()
        workflow = InitRepo(TEST_REPO, lm, json_store)
        workflow.run()

        # Verify config was created and stored
        expected_config = RepoConfig(
            repo_name="JohnPeng47/TestRepo",
            url="git@github.com:JohnPeng47/TestRepo.git",
            source_folder=str(TEST_REPO.resolve())
        )
        
        stored_config = json_store.get_repo_config("JohnPeng47/TestRepo")
        assert stored_config == expected_config
