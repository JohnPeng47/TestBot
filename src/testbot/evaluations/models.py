from typing import Dict, Callable, List
from dataclasses import dataclass

@dataclass
class EvalData:
    name: str
    dataset: Dict
    eval_fn: Callable
    score_fns: List[Callable]