from testbot.llm import LLMModel
from testbot.models import RepoConfig, TestFileData
from testbot.store import TestBotStore
from testbot.code import SupportedLangs, EXTENSIONS, TEST_PATTERNS

import fnmatch
from collections import defaultdict
from pathlib import Path
import git

from .lmp import IdentifyModules, Modules
from ..base import WorkFlow

EXCLUDE_PATTERNS = [
    ".venv/*",
    ".env/*",
    ".build/*",
    ".dist/*"
]

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
        src_test_mapping = defaultdict(list)
        
        # NOTE: handling multiple root paths?
        root_path = self._repo_path
        processed_files = 0
        # NOTE: the module detection code below *should* work for all languages
        # if we also normalize the module path to be in dotted notation ie. mod_a.mod_b.mod_c
        # for f in self._repo_path.rglob(f"*.{EXTENSIONS[lang]}"):
        for f in self._repo_path.rglob(f"*"):
            if any(fnmatch.fnmatch(f, p) for p in EXCLUDE_PATTERNS):
                continue
            if any(f.match(p) for p in TEST_PATTERNS[lang]):
                target_files = []

                print(f"[*] Resolving target source for {str(f)}")

                test_content = open(f, "r").read()
                modules = IdentifyModules().invoke(
                    self._lm,
                    model_name = "claude",
                    test_file = test_content
                )

                # BUG: this loop adds files sys modules and resolves them to self._repo_path
                for module in modules.module_names:
                    rel_mod_path = Path(*module.split("."))
                    mod_path = root_path / (str(rel_mod_path) + EXTENSIONS[lang])

                    if not mod_path.exists():
                        # use matching filename to find the actual root pat
                        for source_path in self._repo_path.rglob(f"**/{mod_path.name}"):
                            root_path = Path(*[p for p in source_path.parts][:-1])
                            mod_path = root_path / mod_path.name

                            # print("New root path: ", root_path)
                            # print("New module path: ", mod_path)
                            if not mod_path.exists():
                                raise Exception()

                    print("> found covered src file: ", mod_path)

                    src_test_mapping[str(mod_path.resolve())].append(str(f))
                    target_files.append(mod_path)

                if target_files:
                    print("Saving testfilepath: ", str(f.resolve()))
                    print("Saving targeted files: ", [str(tf.resolve()) for tf in target_files])
                    self._store.update_or_create_testfile_data(
                        TestFileData(
                            id=str(f),
                            name=f.name,
                            filepath=str(f.resolve()),
                            targeted_files=[str(tf.resolve()) for tf in target_files],
                        )
                    )
            processed_files += 1
            if self._limit and processed_files > self._limit:
                print("Limit hit exiting...")
                break

        # TODO: write an integration test for this
        return src_test_mapping