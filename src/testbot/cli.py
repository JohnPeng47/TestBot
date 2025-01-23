import click
from testbot.store import JsonStore, RepoConfig
from testbot.workflow.init.core import InitRepo



@click.group()
def cli():
    """TestBot CLI tool for managing test repositories"""
    pass


@cli.group()
def repo():
    """Commands for managing test repositories"""
    pass


@repo.command()
def init():
    """Initialize a new test repository"""
    store = JsonStore()
    workflow = InitRepo(None, store)
    workflow.run()


# @repo.command()
# def delete():
#     """Delete an existing test repository"""
#     store = JsonStore()
#     # TODO: Implement repository deletion


def main():
    cli()