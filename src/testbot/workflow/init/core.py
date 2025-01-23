from testbot.llm import LLMModel
from testbot.models import RepoConfig
from testbot.store import TestBotStore
from testbot.code import SupportedLangs

from pathlib import Path
import git

from ..base import WorkFlow

def extract_repo(giturl: str) -> str:
    """gitURL -> owner/repo"""
    if giturl.startswith("git@"):
        # SSH format
        path = giturl.split(":")[-1]
    else:
        # HTTPS format 
        path = giturl.split("/")[-2:]
        path = "/".join(path)
        
    return path.replace(".git", "")


class InitRepo(WorkFlow):
    def __init__(self,
                 repo_path: Path,
                 lm: LLMModel,
                 store: TestBotStore):
        super().__init__(lm, store)

        self._repo_path = repo_path

    def _create_repo_config(self) -> RepoConfig:
        """Create and save a repository configuration"""
        repo = git.Repo(self._repo_path)
        remote_url = repo.remotes.origin.url
        repo_name = extract_repo(remote_url)

        config = RepoConfig(
            repo_name=repo_name,
            url=remote_url,
            source_folder=str(self._repo_path.resolve())
        )

        self._store.create_repoconfig(config)
        return config
        
    def run(self):
        self._create_repo_config()
        
