from testbot.llm.llm import LLMModel, LMP
from pydantic import BaseModel
from typing import List

class Modules(BaseModel):
    reasoning: str
    is_single_module: bool
    module_names: List[str]
    
# NOTE: to modify prompt to handle other notations we can ask return in python dotted notation
# to standardize module parsing
class IdentifyModules(LMP):
    prompt = """
{{test_file}}

This is the definition of a module: a single file where imports are source:
from a.b import (
    c,
    d,r
    e
)

a.b is a single module, rathe than a.b.c, a.b.d, a.b.e individually

Is there a single, primary module under test here? If yes list the module. 
(Note that it can be the case that multiple modules are used in the test but there may be a single module that is the main focus of the test)
If not, list all of the important modules under test
Give your reasoning first before responding
"""
    response_format = Modules