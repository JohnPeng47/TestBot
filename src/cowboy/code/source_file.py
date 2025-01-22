from typing import List, Optional, Tuple
from pathlib import Path
from difflib import unified_diff
from dataclasses import dataclass
from enum import Enum

from .parser.python import PythonAST, Indentation
from .langs import SupportedLangs

from src.lib.utils import locate_python_interpreter

from logging import getLogger
from copy import deepcopy

import subprocess
import ast

logger = getLogger("test_results")
longterm_logger = getLogger("longterm")


PARSERS = {
    SupportedLangs.Python: PythonAST,
}

class LintException(Exception):
    pass


class SameNodeException(Exception):
    pass

class NodeNotFound(Exception):
    pass


@dataclass
class Argument:
    name: str
    type: Optional[str]

    def __str__(self):
        name = self.name
        type = f"{':' + self.type if self.type else ''}"
        return name + type

class ASTNode:
    def __init__(
        self,
        name: str,
        range: Tuple[int, int],
        scope: Optional["ASTNode"],
        decorators: List["Decorator"],
        lines: List[str],
        ast_node: Optional[ast.AST],
        is_test: bool = False,
        node_type: Optional["NodeType"] = None,
    ):
        if decorators:
            assert isinstance(decorators[0], Decorator)
        else:
            assert isinstance(decorators, list) and len(decorators) == 0

        self._name = name
        # REFACTOR-AST: create a lang specific AST node class to account for decorators?
        # ie. PyAST/GolangASTc
        self.decorators = decorators
        self.range = self._set_range(range)
        self.lines = [l.rstrip() for l in lines]
        self.is_test = is_test
        self.scope = scope
        self.ast_node = ast_node
        self.node_type = node_type if node_type else self.get_node_type()

    def get_node_type(self):
        return NodeType(self.__class__.__name__)

    def is_ast(self, other_ast: ast.AST) -> bool:
        """
        Checks if self matches another ast.AST node
        """
        return self.ast_node == other_ast

    def __eq__(self, other: object):
        if not isinstance(other, ASTNode):
            return NotImplemented

        return self.name + str(self.range) == other.name + str(other.range)

    def __hash__(self):
        # return sum([ord(c) for c in self.name + str(self.range)])
        return sum([ord(c) for c in self.name + str(self.range)])

    def _set_range(self, range) -> Tuple[int, int]:
        start = self.decorators[0].range[0] if self.decorators else range[0]
        end = range[1]
        return (start, end)

    def set_is_test(self, is_test: bool):
        self.is_test = is_test

    def to_code(self):
        repr = ""
        for dec in self.decorators:
            repr += dec.to_code()

        # newline if we have decorators
        repr += "\n" if repr else ""
        repr += "\n".join([l.rstrip() for l in self.lines])
        return repr

    @property
    def type(self) -> "NodeType":
        return NodeType(self.__class__.__name__)

    @property
    def name(self):
        raise NotImplementedError

    def to_json(self):
        return {
            "name": self._name,
            "range": self.range,
            "lines": self.lines,
            "decorators": [dec.to_json() for dec in self.decorators],
            "is_test": self.is_test,
            "node_type": self.node_type
        }
    
    @classmethod
    def from_json(cls, data):
        return cls(
            name=data["name"],
            scope=None, # initialized later in Function
            range=tuple(data["range"]),
            lines=data["lines"],
            decorators=[Decorator.from_json(dec) for dec in data["decorators"]],
            ast_node=None,  # ast_node isn't serializable
            is_test=data.get("is_test", False),
            node_type=data["node_type"]
        )


@dataclass
class Decorator(ASTNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return self._name
    
class Class(ASTNode):
    def __init__(self, *args, **kwargs):
        self.functions: List[Function] = kwargs.pop("functions", [])
        super().__init__(*args, **kwargs)

    def add_func(self, func: "Function"):
        self.functions.append(func)

    def __eq__(self, other: object):
        if not isinstance(other, Class):
            return NotImplemented
        
        return self._name == other._name

    @property
    def name(self):
        return self._name

    def to_json(self):
        data = super().to_json()
        data.update({
            "functions": [func.to_json() for func in self.functions],
        })
        return data
    
    @classmethod
    def from_json(cls, data):
        functions = [Function.from_json(func) for func in data["functions"]]
        instance = super().from_json(data)
        instance.node_type = NodeType.Class
        instance.functions = functions
        return instance

# need this to prevent infinite recursion with to/from json
@dataclass
class FakeClassScope:
    name: str

class Function(ASTNode):
    def __init__(self, *args, **kwargs):
        # should replace this with ... a prototype that contains .name property
        # self.scope: Class = kwargs.pop("scope", [])
        self.arguments = kwargs.pop("arguments", [])

        super().__init__(*args, **kwargs)

        self.is_test = True if self._name.startswith("test") else False

    def is_meth(self):
        return bool(self.scope)

    def __str__(self):
        return f"{self._name}({', '.join([arg.__str__() for arg in self.arguments])})"

    def is_method(self):
        return self.scope is not None

    def func_name(self):
        return self._name.split(".")[-1]
    
    def to_json(self):
        data = super().to_json()
        data.update({
            "arguments": [arg.__dict__ for arg in self.arguments],
            "scope": {"name": self.scope.name} if self.scope else None
        })
        return data
    
    @classmethod
    def from_json(cls, data):
        arguments = [Argument(**arg) for arg in data["arguments"]]
        instance = super().from_json(data)
        instance.node_type = NodeType.Function
        instance.arguments = arguments
        if data.get("scope", None):
            instance.scope = FakeClassScope(name=data["scope"]["name"])
        return instance

    @property
    def name(self):
        scope_prefix = f"{self.scope.name}." if self.scope else ""
        return f"{scope_prefix}{self._name}"


class NodeType(str, Enum):
    Function = Function.__name__
    Class = Class.__name__
    Decorator = Decorator.__name__


class SourceFile:
    def __init__(
        self,
        lines: List[str],
        path: Path,
        language: str = "python",
    ):
        self._path = path
        self._lang = language

        self.functions: List[Function] = []
        self.classes: List[Class] = []
        self.lines: List[str] = []
        self.indentation: Indentation = None

        self.update_file_state(lines)

    def clone(self) -> "SourceFile":
        return deepcopy(self)

    @property
    def path(self):
        return self._path

    def update_file_state(self, lines: List[str]):
        """
        Updates instance variables
        """
        assert isinstance(lines, list)
        
        # NEWTODO(Runner): replace PythonAST with generic lang interface that parses function/classes
        self.ast_parser = PARSERS[self._lang]("\n".join(lines))
        self.functions, self.classes, self.indentation = self.ast_parser.parse()
        self.lines = lines

    def __repr__(self):
        return f"{self.path}"

    def diff(self, other: "SourceFile"):
        if not isinstance(other, SourceFile):
            raise TypeError("Can only diff SourceFile instances")

        a = self.to_code().splitlines(keepends=True)
        b = other.to_code().splitlines(keepends=True)
        diff = "".join(unified_diff(a, b))

        return diff

    # this would be another easy test for modification in test_parsing
    # commit: c58ded2
    def find_class(self, class_name: str) -> Optional[Class]:
        """
        Finds a class by name
        """
        return self.find_by_nodetype(class_name, node_type=NodeType.Class)

    def find_function(self, function_name: str) -> Optional[Function]:
        """
        Finds a function by name
        """
        return self.find_by_nodetype(function_name, node_type=NodeType.Function)

    def find_by_nodetype(
        self, node_name: str, node_type: NodeType = NodeType.Function
    ) -> Optional[List[ASTNode]]:
        """
        Finds a function or class by name
        """
        assert type(node_name) == str

        all = [f for f in self.functions + self.classes if f.name == node_name]
        filtered = [f for f in all if f.type == node_type]

        if len(filtered) > 1:
            raise SameNodeException(
                "More than one node found with the same name: ", node_name
            )
        elif len(filtered) == 0:
            raise NodeNotFound(
                "No node found with the given name, did you forget to put NODETYPE param again you dumbass?: ",
                node_name,
                node_type,
                self.path,
            )

        return filtered[0]

    def find_indent(self, lines: List[str]) -> int:
        """
        Find the indentation level of the first non-empty line
        Returns the number of indentation levels (1 level = 4 spaces)
        """
        first_line = next(line for line in lines if line.strip())
        indent = (len(first_line) - len(first_line.lstrip())) / self.indentation.size

        assert indent.is_integer()
        
        return int(indent)

    # NEWTODO: this is not gonna work for other languages .. ie. Java with closing braces
    def append(self, lines: str, class_name: Optional[str] = None) -> None:
        """
        Appends lines to the test file or to an existing class
        """
        if class_name:
            class_node = self.find_by_nodetype(class_name, node_type=NodeType.Class)
            _, end = class_node.range
        else:
            end = len(self.lines) - 1
        
        lines = lines.split("\n")
        first_line = next(line for line in lines if line.strip()) # first non-empty line
        # we essentially have two cases to handle indents
        # 1. we are adding to the global scope, no indents
        # 2. we are adding to a class, we need to indent the lines
        assert len(first_line) == len(first_line.lstrip())
        
        if class_name:
            class_indent = self.find_indent(class_node.lines)
            diff_indent = class_indent + 1
            lines = [diff_indent * self.indentation.char * self.indentation.size + l for l in lines]
        
        lines = self.lines[: end + 1] + lines + self.lines[end + 1 :]
        self.update_file_state(self.to_linted_code(lines).split("\n"))

    # NEWTODO(Tests): tests need to be written for this class
    def delete(self, node_name: str, node_type: NodeType = NodeType.Function) -> None:
        """
        Deletes a new function or class to the file, and updates SourceFile instance accordingly without hitting file
        """
        node = self.find_by_nodetype(node_name, node_type=node_type)
        start, end = node.range

        longterm_logger.info(f"Deleting: {node.name} => {start} to {end}")

        lines = self.lines[:start] + self.lines[end + 1 :]

        if node_type == NodeType.Function:
            self.functions = [f for f in self.functions if f.name != node_name]
        elif node_type == NodeType.Class:
            self.classes = [c for c in self.classes if c.name != node_name]

        self.update_file_state(self.to_linted_code(lines).split("\n"))

    def map_line_to_node(
        self, start: int, end: int
    ) -> Optional[Tuple[ASTNode, ASTNode]]:
        """
        Finds the function and/or class that contains the line
        """
        for node in self.functions:
            if node.range[0] <= start and end <= node.range[1]:
                return node, node.scope
        return None, None

    def to_code(self) -> str:
        """
        Converts the sourcefile to code
        """

        return "\n".join(self.lines)

    def to_num_lines(self) -> str:
        return "\n".join([f"{i}: {line}" for i, line in enumerate(self.lines)])

    def to_llm_repr(self) -> str:
        repr_str = ""
        repr_str += "\n".join([f.__str__() for f in self.functions if not f.scope])
        repr_str += "\n"
        repr_str += "\n".join([c.__str__() for c in self.classes])
        return repr_str

    def to_json(self):
        return {
            "path": str(self.path),
            "lines": self.lines
        }
    
    @classmethod
    def from_json(cls, data):
        return cls(
            lines=data["lines"],
            path=Path(data["path"])
        )

    def aider_rep(self) -> str:
        """
        Returns a string representation of the file with class/function structure,
        using vertical bars and ellipses for visual hierarchy.
        """        
        output = [f"{self.path}:"]
        output.append("⋮...")
                
        # we only print funcs from the global scope
        global_funcs = [func for func in self.functions if not func.scope]
        all_nodes = sorted(self.classes + global_funcs, key=lambda x: x.range[0])
        
        for node in all_nodes:
            # Skip nested nodes as they'll be handled by their parents
            if node.scope is not None:
                continue
            
            if node.type == "Class":
                # Add class definition
                line = f"│class {node.name}"
                output.append(line + ":")
                
                # Add empty line if class has attributes
                if node.functions:
                    output.append("│")
                
                # Add nested functions with indentation
                for func in sorted(node.functions, key=lambda x: x.range[0]):
                    # Add decorators
                    for dec in func.decorators:
                        output.append(f"│\t@{dec.name}")
                    # Add function definition
                    func_name = func.name.split(".")[1]
                    args_str = ", ".join(arg.name for arg in func.arguments)
                    output.append(f"│\tdef {func_name}({args_str}):")
                    output.append("⋮...")
                
            else:  # Function
                # Add decorators
                for dec in node.decorators:
                    output.append(f"│@{dec.name}")
                # Add function definition
                args_str = ", ".join(arg.name for arg in node.arguments)
                output.append(f"│def {node.name}({args_str}):")
                output.append("⋮...")
        
        return "\n".join(output)

class TestFile(SourceFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def test_nodes(self) -> List[ASTNode]:
        return [n for n in self.functions + self.classes if n.is_test]

    def test_funcs(self) -> List[Function]:
        return [func for func in self.functions if func.is_test]

    def test_classes(self) -> List[Class]:
        return [c for c in self.classes if c.is_test]

    def __repr__(self):
        return f"{self.path}"

    def diff_test_funcs(self, new_file: "TestFile") -> List[Function]:
        """
        Returns a list of nodes that are in the new file but not in the current file
        """
        assert Path(self._path) == Path(new_file.path)

        return [
            f
            for f in new_file.test_funcs()
            if f.name not in [my_f.name for my_f in self.test_funcs()]
        ]    
    