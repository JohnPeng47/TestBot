from sqlmodel import SQLModel, create_engine
from testbot.evaluations.models import Commit  # Import your models

class EvalStore:
    def __init__(self, sqlite_path: str = "./eval.db"):
        # Create SQLite URL (prefix sqlite:/// is required)
        sqlite_url = f"sqlite:///{sqlite_path}"

        self.engine = create_engine(sqlite_url, echo=False)  # Set echo=True for SQL debugging
        SQLModel.metadata.create_all(self.engine)

    def upsert_commit(self, Commit: Commit):
        with self.engine.begin() as session:
            session.add(Commit)
            session.commit()