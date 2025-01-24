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
    
    def _merge_code(self, new_code: str, target_code: str) -> str:
        """Apply the code sandwich (context + new code) to the target code file.
        
        Args:
            code_sandwhich: Output from LLM containing context lines and new code
            target_code: Original test file content
            
        Returns:
            Updated test file content with new code inserted
        """
        # Split the sandwich into parts
        parts = new_code.split("\n")
        
        # Find the context and new code
        context_before = []
        new_code_lines = []
        found_marker = False
        
        for line in parts:
            if "NEW CODE ALERT" in line:
                found_marker = True
                continue
                
            if not found_marker:
                context_before.append(line)
            else:
                new_code_lines.append(line)
        
        # Find insertion point using the before context
        if context_before:
            context_str = "\n".join(context_before)
            insert_pos = target_code.find(context_str)
            if insert_pos != -1:
                insert_pos += len(context_str)
                return (
                    target_code[:insert_pos] + 
                    "\n" + "\n".join(new_code_lines) + "\n" +
                    target_code[insert_pos:]
                )
        
        # If context not found or no context provided, append to end
        return target_code + "\n" + "\n".join(new_code_lines)

    # Current(V1): only making use of changed code (non-test) files
    # TODO(V2): make use of the existing test mappings to modify test cases
    # TODO(V3): currently ignoring test files, but we need to handle special case
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
            if op == RecommendedOp.NO_ACTION:
                continue

            # not handling modify test cases for now
            if op == RecommendedOp.NEW_TESTCASE:
                test_files = self._store.get_testfiles_from_srcfile(src_file)
                src_and_test.append((src_file, test_files))

        # TODO: do this in parallel
        for src_file, test_files in src_and_test:
            with open(test_files, "r") as f:
                existing_tests = f.read()

            new_test = GenerateTestWithExisting().invoke(
                self._lm,
                model_name = "claude",
                patch = str(self._commit.find_diff(src_file)),
                source_file = src_file,
                existing_tests = existing_tests
            )
            new_testfile = self._merge_code(new_test.code, existing_tests)