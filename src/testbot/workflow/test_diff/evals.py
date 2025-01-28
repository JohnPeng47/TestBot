from typing import Dict, List
from pathlib import Path

from src.testbot.evaluations.models import EvalData

from .lmp import FilterCommitFilesBatchedV1, FilteredSrcFiles, RecommendedOp

DATASET = [
    {
        "input": {
            "prompt_args": {
                "patch": (
                    "diff --git a/a.py b/a.py\n"
                    "index 0d1bea4..9eb692f 100644\n"
                    "--- a/a.py\n"
                    "+++ b/a.py\n"
                    "@@ -10,4 +10,7 @@ def divide(a, b):\n"
                    "    \"\"\"Divide a by b.\"\"\"\n"
                    "    if b == 0:\n"
                    "        raise ValueError(\"Cannot divide by zero\")\n"
                    "-    return a / b\\ No newline at end of file\n"
                    "+    return a / b\n"
                    "+def average(**args):\n"
                    "+    total = sum(args)\n"
                    "+    return total / len(args)   \n"
                ),
                "src_and_test": [(
                    "/home/ubuntu/cowboy-local/tests/test_repos/test_repo/a.py",
                    (
                        "/home/ubuntu/cowboy-local/tests/test_repos/test_repo/tests/test_a.py",
                        """import pytest
    from a import hello, multiply, divide

    def test_hello():
        assert hello(2, 3) == 5
        assert hello(-1, 1) == 0
        assert hello(0, 0) == 0

    def test_multiply():
        assert multiply(2, 3) == 6
        assert multiply(-2, 3) == -6
        assert multiply(0, 5) == 0

    def test_divide():
        assert divide(6, 2) == 3
        assert divide(5, 2) == 2.5
        assert divide(-6, 2) == -3

    def test_divide_by_zero():
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(5, 0)"""
                    )
                )],
                "repo_path": "/home/ubuntu/cowboy-local/tests/test_repos/test_repo"
            }
        },
        "expected": [(
            "a.py", 
            RecommendedOp.NEW_TESTCASE
        )]
    }
]

# CONSISTENCY
def eval_consistency(data: Dict) -> Dict:
    run_args = data["run_args"]
    prompt_args = data["prompt_args"]

    # return {
    #     "result" : [
    #         FilterCommitFilesBatchedV1(
    #             run_args["model"],
    #             model_name = run_args["model_name"],
    #             patch = prompt_args["patch"],
    #             src_and_test = prompt_args["src_and_test"],
    #             repo_path = Path(prompt_args["repo_path"])
    #         ) 
    #         for _ in range(run_args["iters"])
    #     ]
    # }
    
    result = []
    for _ in range(run_args["iters"]):
        filtered = FilterCommitFilesBatchedV1().invoke(
            run_args["model"],
            model_name = run_args["model_name"],
            patch = prompt_args["patch"],
            src_and_test = prompt_args["src_and_test"],
            repo_path = Path(prompt_args["repo_path"])
        )
        result.append(filtered)

    return {"result": result}

def score_consistency(output: Dict, expected: Dict) -> float:
    filtered_files: List[FilteredSrcFiles] = output["result"]
    
    total_matched = len(filtered_files) * len(expected)
    actual_matched = 0

    for res in filtered_files:
        for f1, op1 in res.file_and_op:
            for f2, op2 in expected:
                print("[ACTUAL]: ", f1, op1)
                print("[EXPECTED]: ", f2, op2)
                print(f1 == f2, op1 == op2)

                if f1 == f2 and op1 == op2:
                    actual_matched += 1
                    break
    
    return actual_matched / total_matched

FILTER_SRC_FILES_CONSISTENCY = EvalData(
    "filtersrc_consistency",
    DATASET,
    eval_consistency,
    [score_consistency]
)