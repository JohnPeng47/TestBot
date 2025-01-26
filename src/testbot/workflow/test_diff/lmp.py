from testbot.llm import (
    LMP, 
    CodeEdit, 
    LLMVerificationError
)
from pydantic import BaseModel
from typing import List, Tuple
from enum import Enum

class RecommendedOp(Enum):
    NEW_TESTCASE = "NEW_TESTCASE"
    MODIFY_TESTCASE = "MODIFY_TESTCASE"
    NO_ACTION = "NO_ACTION"

class FilteredSrcFiles(BaseModel):
    file_and_op: List[Tuple[str, RecommendedOp]]

# NOTE: ideally we can get *block level* src -> test mappings so we can better fit into prompt
# instead of jamming all test case
class FilterCommitFilesBatchedV1(LMP):
    """Filter out all changed files in single commit"""

    prompt = """
{{commit}}

Changed source files: {{code_files}}

Given the source files that changed above, which of these changed files should be considered
for generating a unit test case for or modifying an existing test case? Your response should be 
a list of tuples in the form (source_file, RECOMMENDED_OP), where RECOMMENDED_OP is one of:

- NEW_TESTCASE: a new, *non-trivial* feature is introduced that demands a test coverage
- MODIFY_TESTCASE: changes/updates to an existing feature, that warrants a modification in coverage
- NO_ACTION: trivial changes that does not warrant a unit test
"""
    response_format = FilteredSrcFiles

    def _verify_or_raise(self, res: FilteredSrcFiles, commit: str, code_files: List[str]):
        srcfiles_in_res = list(filter(lambda f: f[0] in code_files, res.file_and_op))

        if len(srcfiles_in_res) != len(code_files):
            raise LLMVerificationError("Response does not contain all changed source files")

# PROMPTDESIGN: potentially combine this and the other into a single prompt
# DESIGN: we should log the patch and then the generated file (minus before_context)
class GenerateTestWithExisting(CodeEdit):
    prompt = """
TASK:
Patch:
{{patch}}

Source file: 
{{source_filename}}

Existing test cases (APPEND_TARGET):
{{existing_tests}}

Write a new test case for the newly added code in the patch
"""