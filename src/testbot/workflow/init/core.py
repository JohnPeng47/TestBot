from testbot.llm import LLMModel
from testbot.models import RepoConfig, TestFileData
from testbot.store import TestBotStore
from testbot.code import SupportedLangs, EXTENSIONS, TEST_PATTERNS

from pathlib import Path
import git

from .lmp import IdentifyModules, Modules
from ..base import WorkFlow

class LanguageNotSupported(Exception):
    """Raised when the language of the repo is not supported"""


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
                 store: TestBotStore,
                 language: str = "",
                 limit: int = None):
        super().__init__(lm, store)

        self._repo_path = repo_path
        self._language = language
        self._limit = limit

    def _create_repo_config(self) -> RepoConfig:
        """Create and save a repository configuration"""
        repo = git.Repo(self._repo_path)
        remote_url = repo.remotes.origin.url
        repo_name = extract_repo(remote_url)

        if self._language and self._language not in SupportedLangs:
            raise LanguageNotSupported(f"Language {self._language} is not supported")

        config = RepoConfig(
            repo_name=repo_name,
            url=remote_url,
            source_folder=str(self._repo_path.resolve()),
            language=self._language
        )

        self._store.create_repoconfig(config)
        return config
    
    def _identify_language(self) -> SupportedLangs:
        """Identify a supported language of the repo and throws error if no language is supported"""
        extension_counts = {}
        reverse_extensions = {v: k for k, v in EXTENSIONS.items()}

        for f in self._repo_path.rglob("*"):
            if f.suffix in EXTENSIONS.values():
                extension_counts[f.suffix] = extension_counts.get(f.suffix, 0) + 1

        if not extension_counts:
            raise LanguageNotSupported(f"Could not detect lang. Is it one of the supported: {list(SupportedLangs)}")
            
        most_common_ext = max(extension_counts.items(), key=lambda x: x[1])[0]
        return reverse_extensions[most_common_ext]    
    
    def run(self):
        self._create_repo_config()
        lang = self._identify_language()

        print("Detected lang: ", lang)
        
        root_path = self._repo_path
        for f in self._repo_path.rglob("*"):
            if any(f.match(p) for p in TEST_PATTERNS[lang]):
                target_files = []

                print(f"Resolving target source for {str(f)}")

                test_content = open(f, "r").read()
                modules = IdentifyModules().invoke(
                    self._lm,
                    model_name = "claude",
                    test_file = test_content
                )

                for module in modules.module_names:
                    print("Module: ", module)
                    mod_path = Path(*module.split("."))
                    full_path = root_path / (str(mod_path) + EXTENSIONS[lang])

                    print("Module path: ", full_path)

                    if not full_path.exists():
                        # TODO: add handling for this case
                        raise Exception("WTF!")
                        continue

                    target_files.append(str(full_path.resolve()))

                if target_files:
                    self._store.update_or_create_testfile_data(
                        TestFileData(
                            id=str(f),
                            name=f.name,
                            filepath=str(f),
                            targeted_files=target_files,
                            # test_metadata={"type": "unit"}
                        )
                    )
