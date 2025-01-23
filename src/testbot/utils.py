import functools
import random
import string
import uuid
import time
import functools
import configparser
from pathlib import Path
from colorama import Fore, Style
from pathlib import Path
from typing import List, Set
import os

# support for all strings as multi-line
import yaml


DELIMETER = "================================================================================"

def str_presenter(dumper, data):
    """Configure yaml for dumping multiline strings."""
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data.strip(), style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter)


# nested level get() function
def resolve_attr(obj, attr, default=None):
    """Attempts to access attr via dotted notation, returns none if attr does not exist."""
    try:
        return functools.reduce(getattr, attr.split("."), obj)
    except AttributeError:
        return default


def gen_random_name():
    """
    Generates a random name using ASCII, 8 characters in length
    """

    return "".join(random.choices(string.ascii_lowercase, k=8))


def generate_id():
    """
    Generates a random UUID
    """
    return str(uuid.uuid4())


def async_timed(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Function {func.__name__} took {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper

def confirm_action(prompt: str) -> bool:
    """Get user confirmation for an action"""
    prompt += "\n(y/N)"
    
    while True:
        response = input(f"{prompt}").lower().strip()
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no', '']:
            return False
        print("Please answer 'y' or 'n'")

def sync_timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Function {func.__name__} took {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper

def cyan_text(text: str) -> str:
    """Returns text in red color using colorama"""
    return f"{Fore.CYAN}{text}{Style.RESET_ALL}"
    
def green_text(text: str) -> str:
    """Returns text in green color using colorama"""
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"

def red_text(text: str) -> str:
    """Returns text in red color using colorama"""
    return f"{Fore.RED}{text}{Style.RESET_ALL}"

def dim_text(text: str) -> str:
    """Returns text in dim color using colorama"""
    return f"{Style.DIM}{text}{Style.RESET_ALL}"

def load_env(env_path: str = ".env") -> None:
    """Load environment variables from a .env file into os.environ.
    
    Args:
        env_path: Path to the .env file. Defaults to ".env" in current directory.
    """
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip("\"'")
    except FileNotFoundError:
        print(f"Warning: {env_path} file not found")
