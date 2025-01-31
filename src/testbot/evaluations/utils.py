import click
from sqlmodel import Session
from functools import wraps

def get_db_session(ctx: click.Context) -> Session:
    """Helper to get session from context"""
    return Session(ctx.obj["engine"])

# Create a decorator for commands that need the db session
def with_db_session(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        ctx = click.get_current_context()
        with get_db_session(ctx) as session:
            return f(*args, session=session, **kwargs)
    return wrapper
