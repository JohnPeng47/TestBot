from typing import Any, Dict, Optional, Type, List, Tuple
import xml.etree.ElementTree as ET
import inspect
from pathlib import Path
import sqlite3
import hashlib
import json
import functools
import time
import jinja2
import re

from pydantic import BaseModel
import tiktoken

import instructor
from litellm import completion
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from testbot.utils import green_text

client = instructor.from_litellm(completion)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens

SHORT_NAMES = {
    "gpt-4o" : "gpt-4o",
    "claude" : "claude-3-5-sonnet-20240620",
    "deepseek" : "deepseek/deepseek-chat"
}

class LLMVerificationError(Exception):
    pass

class ChatMessage(BaseModel):
    role: str
    content: str

class LLMModel:
    """
    LLMModel that wraps LiteLLM and Instructor for structured output
    Also implements a caching layer
    """
    
    def __init__(
        self,
        use_cache: bool = False,
        configpath: Path = Path(__file__).parent / "cache.yaml",
        dbpath: Path = Path(__file__).parent / "llm_cache.db"
    ) -> None:
        """
        Initialize the LLM model with the specified provider and configuration.
        
        Args:
            provider: The name of the model provider (e.g., "openai", "anthropic")
        """
        self.use_cache = use_cache
        self.config = self._read_config(configpath)

        # Initialize cache-related attributes
        self.cache_enabled_functions = self._get_cache_enabled_functions()
        print("Enabled functions: ", "\n".join([f for f, enabled in self.cache_enabled_functions.items() if enabled]))

        self.db_connection = None
        
        # Initialize cache database
        self._initialize_cache(dbpath)
        
        # Add call chain tracking
        self.call_chain = []

    def _read_config(self, fp: Path):
        # with open(fp, "r") as f:
        #     config = yaml.safe_load(f)
        # return config
        # Cache turned off
        return {}

    def _get_caller_info(self):
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back  # Go back one more frame
        caller_function = caller_frame.f_code.co_name
        caller_filename = caller_frame.f_code.co_filename

        return caller_filename, caller_function
        
    def _get_cache_enabled_functions(self) -> Dict[str, bool]:
        """Extract function names and their cache states from config."""
        cache_states = {}
        for func_name, settings in self.config.items():
            # Look for cache setting in the list of dictionaries
            cache_setting = next((item.get('cache') 
                                for item in settings 
                                if isinstance(item, dict) and 'cache' in item), 
                               False)
            cache_states[func_name] = cache_setting
        return cache_states

    def _initialize_cache(self, dbpath: Path) -> None:
        """Initialize SQLite connection and create cache table if it doesn't exist."""
        self.db_connection = sqlite3.connect(dbpath)
        cursor = self.db_connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                function_name TEXT,
                model_name TEXT,
                prompt_hash TEXT,
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (function_name, model_name, prompt_hash)
            )
        """)
        self.db_connection.commit()

    def _hash_prompt(self, prompt: str, model: str, key: int = 0) -> str:
        """Create a consistent hash of the prompt with the key used for iterative prompts."""
        return hashlib.sha256(prompt.encode() + model.encode() + str(key).encode()).hexdigest()

    def _delete_hash(self, function_name: str, model_name: str, prompt_hash: str) -> None:
        """Delete a specific cache entry by its hash."""
        if not self.db_connection:
            return
            
        cursor = self.db_connection.cursor()
        cursor.execute(
            "DELETE FROM llm_cache WHERE function_name = ? AND model_name = ? AND prompt_hash = ?",
            (function_name, model_name, prompt_hash)
        )
        self.db_connection.commit()

    def _get_cached_response(self, function_name: str, model_name: str, prompt_hash: str) -> Optional[str]:
        """Retrieve cached response if it exists."""
        if not self.db_connection:
            return None
            
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT response FROM llm_cache WHERE function_name = ? AND model_name = ? AND prompt_hash = ?",
            (function_name, model_name, prompt_hash)
        )
        result = cursor.fetchone()
        return json.loads(result[0]) if result else None

    def _cache_response(self, function_name: str, model_name: str, prompt_hash: str, response: Any) -> None:
        """Store response in cache."""
        if not self.db_connection:
            return
                    
        cursor = self.db_connection.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO llm_cache (function_name, model_name, prompt_hash, response) VALUES (?, ?, ?, ?)",
            (function_name, model_name, prompt_hash, json.dumps(response))
        )
        self.db_connection.commit()

    def cache_llm_response(func):
        """Method decorator to handle LLM response caching."""
        @functools.wraps(func)
        def wrapper(self, 
                    prompt: str, 
                    *, 
                    model_name: str = "gpt-4o", 
                    response_format: Optional[Type[BaseModel]] = None, 
                    use_cache: Optional[bool] = None,
                    delete_cache: bool = False,
                    key: int = 0,
                    **kwargs):
            
            # Use instance default if use_cache is None
            use_cache = self.use_cache if use_cache is None else use_cache
            
            # Track the call
            caller_filename, caller_function = self._get_caller_info()
            self.call_chain.append((caller_filename, caller_function))
            
            # Check if caching is enabled for this function
            # cache_enabled = (use_cache and 
            #                caller_function in self.cache_enabled_functions and 
            #                self.cache_enabled_functions[caller_function])
            cache_enabled = use_cache
            
            if delete_cache:
                self._delete_hash(caller_function, model_name, self._hash_prompt(prompt, model_name, key=key))
                
            elif not delete_cache and cache_enabled:
                # Check cache for existing response
                prompt_hash = self._hash_prompt(prompt, model_name, key=key)
                cached_response = self._get_cached_response(caller_function, model_name, prompt_hash)
                                
                if cached_response is not None:
                    caller = inspect.stack()[1]  # Get immediate caller
                    print(green_text(f"Returning from cache[LLM]:"))
                    print(green_text(f"|---> Called from {caller.filename}:{caller.lineno} in {caller.function}"))

                    # If response is a Pydantic model, reconstruct it
                    if response_format is not None:
                        return response_format.model_validate(cached_response)
                    
                    return cached_response["content"]
            
            # Get response from LLM
            res = func(self, 
                       prompt, 
                       model_name=model_name, 
                       response_format=response_format, 
                       **kwargs)
            
            # Prepare response for caching
            if isinstance(res, ModelResponse):
                res = res.choices[0].message.content
                cached_response = res
            elif isinstance(res, BaseModel):
                cached_response = res.model_dump()
            else:
                raise Exception(f"Unsupported return type: {type(res)}")
                
            # Cache the response if enabled
            if cache_enabled:
                print("Caching response: ", model_name, prompt[:20], key, prompt_hash[:4])
                self._cache_response(caller_function, model_name, prompt_hash, cached_response)
                
            return res
            
        return wrapper

    @cache_llm_response
    def invoke(self, 
               prompt: str | List[ChatMessage],
               *, 
               model_name: str = "gpt-4o", 
               response_format: Optional[Type[BaseModel]] = None,
               use_cache: bool = True,
               **kwargs) -> Any:
        """Modified invoke method with caching."""

        if isinstance(prompt, str):
            messages = [{
                "role": "user",
                "content": prompt,
            }]
        elif isinstance(prompt, list):
            messages = [m.dict() for m in prompt]
            
        model_name = SHORT_NAMES[model_name]
        res = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_model=response_format,
            **kwargs
        )
        return res
    
    def __del__(self):
        """Cleanup database connections on object destruction."""
        # import traceback
        # print("Cleaning up database connection. Called from:")
        # traceback.print_stack()

        if self.db_connection:
            self.db_connection.close()


# DESIGN: not sure how to enforce this but we should only allow JSON serializable
# args to be passed to the model, to be compatible with Braintrust 
class LMP[T]:
    """
    A language model progsram
    """
    prompt: str
    response_format: Type[T]

    def _prepare_prompt(self, **prompt_args) -> str:
        return jinja2.Template(self.prompt).render(**prompt_args)

    def _verify_or_raise(self, res, **prompt_args):
        return True

    def _process_result(self, res, **prompt_args) -> Any:
        return res

    def invoke(self, 
               model: LLMModel,
               model_name: str = "claude",
               max_retries: int = 3,
               retry_delay: int = 1,
               use_cache: bool = False,
               # gonna have to manually specify the args to pass into model.invoke
               # or do some arg merging shit here
               **prompt_args) -> Any:
        prompt = self._prepare_prompt(**prompt_args)

        current_retry = 1
        while current_retry <= max_retries:
            try:
                res = model.invoke(prompt, 
                                   model_name=model_name,
                                   response_format=self.response_format,
                                   use_cache=use_cache)
                self._verify_or_raise(res, **prompt_args)
                return self._process_result(res, **prompt_args)
            
            except LLMVerificationError as e:
                current_retry += 1
                
                if current_retry > max_retries:
                    raise e
                
                # Exponential backoff: retry_delay * (2 ^ attempt)
                current_delay = retry_delay * (2 ** (current_retry - 1))
                time.sleep(current_delay)
                print(f"Retry attempt {current_retry}/{max_retries} after error: {str(e)}. Waiting {current_delay}s")

class AppendOp(BaseModel):
    targeted_line: str
    new_code: str

    def apply(self, target_code: str) -> str:
        return target_code.replace(self.targeted_line, self.targeted_line + "\n" + self.new_code)
    
class ModifyOp(BaseModel):
    targeted_line: str
    new_code: str

    def apply(self, target_code: str) -> str:
        return target_code.replace(self.targeted_line, self.new_code)

# IMPROVE: problem with append not generating newlines
# for example:
#     with pytest.raises(ValueError, match="Cannot divide by zero"):
#         divide(5, 0)
# def test_average():
#     assert average(1, 2, 3) == 2
class CodeEdit(LMP):    
    KEYWORDS = ["TASK", "APPEND_TARGET"]
    APPEND_CODE_PROMPT = """
Now generate code that implements TASK. The generated code must be added to the file labeled as APPEND_TARGET as a series of edit operations.

REQUIRED STEPS:
1. Generate an ordered series of steps explaining how to accomplish TASK
2. Generate a list of EDIT_OPS that implement those steps

VALID EDIT_OPS:

1. APPEND_OP - Used to add new code after an existing line
   Structure:
   <append_op>
       <target_line>existing line to append after</target_line>
       <new_code>code to be inserted</new_code>
   </append_op>

2. MODIFY_OP - Used to replace an existing line with new code
   Structure:
   <modify_op>
       <target_line>existing line to modify</target_line>
       <new_code>code to replace it with</new_code>
   </modify_op>

NOTE: All operations must be enclosed in <edit_op> tags
NOTE: Make sure to observer proper new lines

NEWLINE EXAMPLES:
    return something
def do_stuff(**args): # BAD: should add a newline in between
    ...


EXAMPLES:

-------- EXAMPLE 1: APPEND OPERATIONS --------

Steps:
1. Import sys module for exit functionality
2. Add sys.exit() call at end of function

<edit_op>
<append_op>
    <target_line>import os</target_line>
    <new_code>import sys</new_code>
</append_op>

<append_op>
    <target_line>    print("Hello, world!")</target_line>
    <new_code>    sys.exit()</new_code>
</append_op>
</edit_op>

-------- EXAMPLE 2: MIXED OPERATIONS --------

Steps:
1. Add logging import
2. Replace print with logging
3. Add logging configuration

<edit_op>
<append_op>
    <target_line>import os</target_line>
    <new_code>import logging</new_code>
</append_op>

<modify_op>
    <target_line>print("Starting application...")</target_line>
    <new_code>logging.info("Starting application...")</new_code>
</modify_op>

<append_op>
    <target_line>import logging</target_line>
    <new_code>logging.basicConfig(level=logging.INFO)</new_code>
</append_op>
</edit_op>

-------- END EXAMPLES --------

RESULTING FILE FORMAT:
The final APPEND_TARGET file should look like this after applying the operations:

import os
import sys

def main():
    print("Hello, world!")
    sys.exit()

Now generate your code:"""
    response_format = None

    def __init__(self, target_code: str = None):
        self._target_code = target_code
        
    def _prepare_prompt(self, **prompt_args) -> str:
        prompt = super()._prepare_prompt(**prompt_args)
        
        for k in self.KEYWORDS:
            if k not in prompt:
                raise ValueError(f"{k} not found in prompt")
        
        final_prompt = prompt + self.APPEND_CODE_PROMPT
        return final_prompt
    
    def _extract_edit_ops(self, content: str) -> List[AppendOp | ModifyOp]:
        """
        Parse edit operations from an XML-formatted string while preserving indentation.
        
        Args:
            content (str): String containing edit operations in XML format
            
        Returns:
            EditOps: Parsed operations in structured format
        """
        # Extract the XML portion
        xml_match = re.search(r'<edit_op>.*?</edit_op>', content, re.DOTALL)
        if not xml_match:
            raise ValueError("No edit_op tags found in content")
        
        xml_content = xml_match.group()
        
        # Parse XML while preserving whitespace
        parser = ET.XMLParser()
        root = ET.fromstring(xml_content, parser=parser)
        edit_ops = []
        
        # Process all append operations
        for append_op in root.findall('.//append_op'):
            append_line = append_op.find('target_line')
            new_code = append_op.find('new_code')
            
            if append_line is not None and new_code is not None:
                # Preserve indentation by getting raw text content
                append_line_text = ''.join(append_line.itertext()).rstrip()
                new_code_text = ''.join(new_code.itertext()).rstrip()
                
                # Remove any common leading newlines
                append_line_text = append_line_text.lstrip('\n')
                new_code_text = new_code_text.lstrip('\n')
                
                edit_ops.append(
                    AppendOp(
                        targeted_line=append_line_text,
                        new_code=new_code_text
                    )
                )
        
        # Process all modify operations
        for modify_op in root.findall('.//modify_op'):
            modify_line = modify_op.find('target_line')
            new_code = modify_op.find('new_code')
            
            if modify_line is not None and new_code is not None:
                # Preserve indentation by getting raw text content
                modify_line_text = ''.join(modify_line.itertext()).rstrip()
                new_code_text = ''.join(new_code.itertext()).rstrip()
                
                # Remove any common leading newlines
                modify_line_text = modify_line_text.lstrip('\n')
                new_code_text = new_code_text.lstrip('\n')
                
                edit_ops.append(
                    ModifyOp(
                        targeted_line=modify_line_text,
                        new_code=new_code_text
                    )
                )
        
        return edit_ops

    def _process_result(self, res: str, **prompt_args) -> str:
        edit_ops = self._extract_edit_ops(res)
        target_code = self._target_code
        for op in edit_ops:
            target_code = op.apply(target_code)
            
        return target_code