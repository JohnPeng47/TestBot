from testbot.code import SupportedLangs
from testbot.llm import LLMModel
from testbot.diff import CommitDiff
from testbot.store.store import TestBotStore

from .lmp import (
    FilterCommitFilesBatchedV1, 
    RecommendedOp, 
    FilteredSrcFiles,
    GenerateTestWithExisting
)
from ..base import WorkFlow

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
        super().__init__(lm, store)
        
        self._repo_name = repo_name
        self._repo_path = repo_path
        self._lang = lang
        self._commit = commit

    # Current(V1): only making use of changed code (non-test) files
    # DESIGN(V2): make use of the existing test mappings to modify test cases
    # DESIGN(V3): currently ignoring test files, but we need to handle special case
    # DESIGN(V4): handle case of unmapped new file
    # where the test file is also included with the commit
    def run(self):
        # first try to filter out useless files
        res: FilteredSrcFiles = FilterCommitFilesBatchedV1().invoke(
            self._lm,
            model_name = "claude",
            commit = str(self._commit),
            code_files = self._commit.code_files,
        )

        src_and_test = []
        for src_file, op in res.file_and_op:
            src_file = self._repo_path / src_file

            if op == RecommendedOp.NO_ACTION:
                continue
            if op == RecommendedOp.NEW_TESTCASE:
                test_files = self._store.get_testfiles_from_srcfile(src_file)
                test_files = [f.filepath for f in test_files]
                src_and_test.append((src_file, test_files))

        # TODO: do this in parallel
        for src_file, test_files in src_and_test:
            # TODO: handle this with another prompt
            with open(test_files[0], "r") as f:
                existing_tests = f.read()

            new_test = GenerateTestWithExisting(target_code=existing_tests).invoke(
                self._lm,   
                model_name = "claude",
                patch = str(self._commit.find_diff(src_file.name)),
                source_file = src_file.name,
                existing_tests = existing_tests
            )
            print("[NEW TESTFILE]: ", new_test)
