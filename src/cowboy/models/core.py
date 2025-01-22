from sqlmodel import SQLModel, Field, JSON
from typing import Optional, List

class TestModuleData(SQLModel, table=True):
    """Stores information about a test module"""
    __tablename__ = "test_modules"

    id: str = Field(default=None, primary_key=True)
    name: str
    filepath: str
    targeted_files: List[str]
    test_metadata: dict = Field(default={}, sa_type=JSON)

class RepoConfig(SQLModel, table=True):
    """Configuration for a GitHub repository"""
    __tablename__ = "repo_configs"

    id: str = Field(default=None, primary_key=True)
    url: str
    source_folder: str
    cloned_folders: Optional[List[str]] = Field(default=[], sa_type=JSON)
    python_conf: dict = Field(sa_type=JSON)

    def __init__(self, **data):
        super().__init__(**data)
        self.url = self.validate_url(self.url)

    @classmethod
    def validate_url(cls, v):
        import re

        if not re.match(r"^https:\/\/github\.com\/[\w-]+\/[\w-]+(\.git)?$", v):
            raise ValueError(
                "URL must be a valid GitHub HTTPS URL and may end with .git"
            )
        if re.match(r"^git@github\.com:[\w-]+\/[\w-]+\.git$", v):
            raise ValueError("SSH URL format is not allowed")
        return v

    def serialize(self) -> dict:
        return {
            "repo_name": self.repo_name,
            "url": self.url,
            "cloned_folders": self.cloned_folders,
            "source_folder": self.source_folder,
            "python_conf": self.python_conf,
            "is_experiment": self.is_experiment
        }

