from testbot.llm.llm import LLMModel
from testbot.diff import CommitDiff
from testbot.store.store import TestBotStore
from testbot.terminal import IO

from testbot.logger import logger as log

from .lmp import (
    FilterCommitFilesBatchedV1, 
    RecommendedOp, 
    FilteredSrcFiles,
    GenerateTestWithExisting,
    CleanUpCode
)
from ..base import WorkFlow

import difflib
from pathlib import Path

class TestDiffWorkflow(WorkFlow):
    """Generates test cases for a code diff"""
    def __init__(self, 
                 commit: CommitDiff,
                 repo_path: Path, 
                 lm: LLMModel,
                 store: TestBotStore,
                 io: IO):
        super().__init__(lm, store)
        
        self._io = io
        self._repo_path = repo_path
        self._commit = commit

    # Current(V1): only making use of changed code (non-test) files
    # IMPROVE(V2): make use of the existing test mappings to modify test cases
    # IMPROVE(V3): currently ignoring test files, but we need to handle special case
    # IMPROVE(V4): handle case of unmapped new file
    # where the test file is also included with the commit
    def run(self):
        log.info(f"Starting TestDiffWorkflow for {self._repo_path}")

        src_and_test = []
        for src_file in self._commit.code_files:
            src_file = self._repo_path / src_file
            src_file = str(src_file.resolve())
            test_files = self._store.get_testfiles_from_srcfile(src_file)
            test_files = [f.filepath for f in test_files]
            
            # TODO: handle case of multiple test files
            log.debug(f"[MAPPED TESTFILES]: {test_files}")
            test_file = test_files[0]
            test_content = open(test_file, "r").read()
            src_and_test.append((src_file, (test_file, test_content)))
    
        # first try to filter out useless files
        # NOTE: currently doing this for single commit all files but might want to
        # split up the commit into separate hunks
        res: FilteredSrcFiles = FilterCommitFilesBatchedV1().invoke(
            self._lm,
            model_name = "claude",
            patch = str(self._commit),
            src_and_test = src_and_test,
            repo_path = self._repo_path
        )

        filtered_src_and_test = []
        for src_file, op in res.file_and_op:
            # actually have a problem here if repo_path is fullpath
            src_file = self._repo_path / src_file
            src_file = src_file.resolve()
            log.info(f"[CHANGED SRCFILE]: {src_file}, {op}\n")

            if op == RecommendedOp.NO_ACTION:
                continue
            if op == RecommendedOp.NEW_TESTCASE:
                test_files = self._store.get_testfiles_from_srcfile(src_file)
                test_files = [f.filepath for f in test_files]
                filtered_src_and_test.append((src_file, test_files))
        
        for src_file, test_files in filtered_src_and_test:
            # TODO: handle filtering files with another prompt
            test_file = test_files[0]

            with open(test_file, "r") as f:
                existing_tests = f.read()

            new_test = GenerateTestWithExisting(target_code=existing_tests).invoke(
                self._lm,   
                model_name = "claude",
                patch = str(self._commit.find_diff(src_file.name)),
                source_file = src_file.name,
                existing_tests = existing_tests
            )

            # TODO: add color here
            diff = difflib.unified_diff(
                existing_tests.splitlines(keepends=True),
                new_test.splitlines(keepends=True),
                fromfile="existing_tests",
                tofile="new_tests"
            )
            diff = "".join(diff)
            if self._io.input(f"{diff}\nWrite changes to {test_file} (y/n)", 
                              validator=lambda res: res.lower() == "y"):
                cleaned_test = CleanUpCode().invoke(
                    self._lm,
                    model_name = "deepseek",
                    code = new_test
                )
                cleaned_test = cleaned_test.code
                log.info(f"[NEW TEST]:\n{cleaned_test}")

                with open(test_file, "w") as f:
                    f.write(cleaned_test)
                
            # create_and_stage_test_diff(self._repo_path, test_file, new_test)
            
