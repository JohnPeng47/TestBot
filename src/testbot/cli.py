import click
from pathlib import Path

from testbot.store import JsonStore
from testbot.workflow.init.core import InitRepo
from testbot.llm import LLMModel
from testbot.utils import load_env


@click.group()
def cli():
    """TestBot CLI tool for managing test repositories"""
    pass


@cli.group()
def repo():
    """Commands for managing test repositories"""
    pass


@repo.command()
@click.argument("repo_path")
@click.option("--language", default=None)
def init(repo_path, language):
    """Initialize a new test repository"""
    store = JsonStore()
    workflow = InitRepo(Path(repo_path), LLMModel(), store, language=language)
    workflow.run()


# @repo.command()
# def delete():
#     """Delete an existing test repository"""
#     store = JsonStore()
#     # TODO: Implement repository deletion

def main():
    load_env() # load LLM API keys

    cli()

if __name__ == "__main__":
    main()