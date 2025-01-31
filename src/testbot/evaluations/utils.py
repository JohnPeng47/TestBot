from testbot.store import TestBotStore

import click
from sqlmodel import Session

def get_db_session_and_store(ctx: click.Context) -> tuple[Session, TestBotStore]:
    """Helper to get session and store from context"""
    return Session(ctx.obj["engine"]), ctx.obj["store"]