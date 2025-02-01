from testbot.llm.llm import LLMModel, LMP
from pydantic import BaseModel
from typing import Any, List
from pathlib import Path

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

    def _process_result(self, res,
                        test_file: str = "",
                        repo_path: Path = None, ) -> List[Path]:
        """Processes module names and attempt to resolve them to actual paths"""
        module_paths = []
        for module in res.module_names:
            rel_mod_path = Path(*module.split("."))
            mod_path = repo_path / str(rel_mod_path)

            if not mod_path.exists():
                # Use matching filename to find actual root path
                for source_path in repo_path.rglob(f"**/{mod_path.name}"):
                    root_path = Path(*[p for p in source_path.parts][:-1])
                    mod_path = root_path / mod_path.name

                    if not mod_path.exists():
                        raise Exception()

            print("> found covered src file: ", mod_path)
            module_paths.append(mod_path)

        return module_paths