from testbot.llm.llm import (
    LMP, 
    CodeEdit, 
    LLMVerificationError
)
from testbot.logger import logger as log

from pathlib import Path
import jinja2
from pydantic import BaseModel
from typing import List, Tuple
from enum import Enum

def is_subpath(subpath: Path, parent_path: Path):
    path_parts = []
    for p in parent_path.parts[::-1]:
        path_parts.append(p)
        if Path(*path_parts) == Path(subpath):
            return True
    return False

class RecommendedOp(Enum):
    NEW_TESTCASE = "NEW_TESTCASE"
    MODIFY_TESTCASE = "MODIFY_TESTCASE"
    NO_ACTION = "NO_ACTION"

class FilteredSrcFiles(BaseModel):
    reason: str
    file_and_op: List[Tuple[str, RecommendedOp]]

class FilterCommitFilesBatchedV1(LMP):
    """Filter out all changed files in single commit"""

    prompt = """
{{patch}}

Here are the list a list of the source files in the commit along with their existing test cases:
{{src_and_test}}

Given the source files that changed above, which of these changed files should be considered
for generating a unit test case for or modifying an existing test case? Your response should be 
a list of tuples in the form (source_file, RECOMMENDED_OP), where RECOMMENDED_OP is one of:

- NEW_TESTCASE: a new, *non-trivial* feature is introduced that is not covered by the existing test cases
- MODIFY_TESTCASE: changes/updates to an existing feature, that warrants a modification in coverage of an existing test case
- NO_ACTION: trivial changes that eddoes not warrant a unit test

Return your output with the stated reason as well as a list of (file, RECOMMENDED_OP) tuples
"""
    response_format = FilteredSrcFiles

    def _prepare_prompt(self,
                        patch: str = "",
                        src_and_test: List[Tuple[str, Tuple[str, str]]] = [],
                        repo_path: Path = None):
        src_and_test_str = ""

        for src_fp, test_file in src_and_test:
            src_fp = Path(src_fp)

            test_fp, test_content = Path(test_file[0]), test_file[1]
            src_and_test_str += f"Source file: {src_fp.relative_to(repo_path)}\n"
            src_and_test_str += f"Existing Test file: {test_fp.relative_to(repo_path)}\n"
            src_and_test_str += f"\n{test_content}\n"
            
        prompt = jinja2.Template(self.prompt).render(patch=patch, src_and_test=src_and_test_str)
        log.info(f"[FILTERED_PROMPT]: {prompt}")
        
        return prompt
            
    def _verify_or_raise(self, 
                         res: FilteredSrcFiles, 
                         patch: str, 
                         src_and_test: List[Tuple[str, Tuple[str, str]]] = [],
                         repo_path: Path = None):
        # Get list of source files from src_and_test input
        src_files = [src for src, _ in src_and_test]
        srcfiles_in_res = list(
            filter(lambda f: any(
                is_subpath(Path(f[0]), (Path(src))) for src in src_files
            ), res.file_and_op
        ))

        if len(srcfiles_in_res) != len(src_files):
            raise LLMVerificationError(f"Response does not contain all source files from src_and_test: \n {srcfiles_in_res}")


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

class CleanedCode(LMP):
    code: str
class CleanUpCode(LMP):
    prompt = """
{{code}}

The code above has been generated by an AI agent. There maybe some formatting issues with the generated code.
Can you please fix any formatting issues such as missing newlines? DO NOT CHANGE the code itself except to fix
formatting issues such as missing/extra newlines
"""
    response_format = CleanedCode