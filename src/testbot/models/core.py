from sqlmodel import SQLModel, Field, JSON
from typing import Optional, List

class TestFileData(SQLModel, table=True):
    """Stores information about a test module"""
    __tablename__ = "test_modules"

    id: str = Field(default=None, primary_key=True)
    name: str
    filepath: str
    targeted_files: List[str] = Field(default={}, sa_type=JSON)
    test_metadata: dict = Field(default={}, sa_type=JSON)

class RepoConfig(SQLModel, table=True):
    """Configuration for a GitHub repository"""
    __tablename__ = "repo_configs"

    id: str = Field(default=None, primary_key=True)
    repo_name: str
    url: str
    source_folder: str
    # cloned_folders: Optional[List[str]] = Field(default=[], sa_type=JSON)

    def __init__(self, **data):
        super().__init__(**data)
        self.url = self.validate_url(self.url)

    @classmethod
    def validate_url(cls, v):
        import re

        if not (re.match(r"^https:\/\/github\.com\/[\w-]+\/[\w-]+(\.git)?$", v) or 
                re.match(r"^git@github\.com:[\w-]+\/[\w-]+\.git$", v)):
            raise ValueError(
                f"{v} must be a valid GitHub HTTPS URL or SSH URL"
            )
        return v