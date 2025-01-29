from braintrust import Eval

from testbot.utils import load_env

load_env()
Eval(
    "testbot",
    data = [{
        "input": 1, 
        "expected": 2
    }],
    task = lambda x: x == 2,
    scores = [lambda x: 1],
    experiment_name="filtersrc_consistency-hello",
    metadata={
        "comments": "This is a test" 
    }
)
