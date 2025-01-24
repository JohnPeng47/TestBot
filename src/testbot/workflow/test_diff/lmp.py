from testbot.llm import LMP, LLMVerificationError
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
# 
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
class GeneratedCode(BaseModel):
    code: str
        
class GenerateTestWithExisting(LMP):
    prompt = """
Patch:
{{patch}}

Source file: 
{{source_file}}

Existing tests:
{{existing_tests}}

Given the patch and the source_file above, extend the existing test cases for the source file.
The way you should generate your new code is as follows:
1. First generate a couple of lines from the existing test, at the point that you plan on inserting your new code
2. Follow this by generating a comment in whatever language the existing tests is in, saying: NEW_CODE_ALERT
3. Then generate your new code

Now generate your code:
"""
    response_format = GeneratedCode

    def _verify_or_raise(self, res: GeneratedCode, **prompt_args):
        existing_tests = prompt_args["existing_tests"]
        parts = res.code.split("\n")
        
        # Find the context and new code
        context_before = []
        found_marker = False
        for line in parts:
            if "NEW_CODE_ALERT" in line:
                found_marker = True
                break

            context_before.append(line)

        code_before = "\n".join(context_before)
        if code_before not in existing_tests:
            raise LLMVerificationError("Context before NEW_CODE_ALERT not found in existing tests")

        if not found_marker:
            raise LLMVerificationError("NEW_CODE_ALERT not found in response")

        if not context_before:
            raise LLMVerificationError("No context found before NEW_CODE_ALERT")