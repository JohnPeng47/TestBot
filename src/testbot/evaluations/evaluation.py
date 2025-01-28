from braintrust import Eval
from typing import Dict
import sys
import argparse

from src.testbot.llm import LLMModel
from src.testbot.config import BRAINTRUST_PROJ_NAME

from .models import EvalData

## import evals
from src.testbot.workflow.test_diff.evals import FILTER_SRC_FILES_CONSISTENCY
from src.testbot.utils import load_env

# there is probably some way to iterate the directory and automagically collect
# all the evals
ALL_EVALS = [
    FILTER_SRC_FILES_CONSISTENCY
]

def eval(eval_data: EvalData,
        model: LLMModel, 
        model_name: str = "claude", 
        experiment_suffix = "",
        comments:str = "",
        iters = 15):
    # add run args to each datum for the eval
    for datum in eval_data.dataset:
        datum["input"]["run_args"] = {
            "model": model,
            "model_name": model_name,
            "iters": iters
        }

    experiment_name = eval_data.name + "_" + model_name + "_" + str(iters) + "-" + experiment_suffix
    metadata = {
        "comments": comments
    }

    return Eval(
        BRAINTRUST_PROJ_NAME,
        eval_data.dataset,
        eval_data.eval_fn,
        eval_data.score_fns,
        experiment_name=experiment_name,
        metadata=metadata
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluations")
    parser.add_argument("eval_name", help="Name of evaluation to run")
    parser.add_argument("--model-name", default="claude", help="Model identifier")
    parser.add_argument("--iters", type=int, default=15, help="Number of iterations to run")
    parser.add_argument("--experiment-suffix", "-s", type=str)
    parser.add_argument("--comments", "-m", type=str)

    args = parser.parse_args()

    load_env()
    model = LLMModel()

    for eval_data in ALL_EVALS:
        if eval_data.name == args.eval_name:
            eval(eval_data, 
                 model, 
                 model_name=args.model_name, 
                 experiment_suffix=args.experiment_suffix, 
                 iters=args.iters)
        
    