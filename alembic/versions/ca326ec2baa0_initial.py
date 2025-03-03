"""initial

Revision ID: ca326ec2baa0
Revises: 
Create Date: 2025-02-01 12:43:29.263840

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = 'ca326ec2baa0'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('commit',
    sa.Column('sha', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('diff', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('repo', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('diff_bytes', sa.Integer(), nullable=False),
    sa.Column('num_files', sa.Integer(), nullable=False),
    sa.Column('num_test_files', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('sha')
    )
    op.create_table('repo_configs',
    sa.Column('id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('repo_name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('url', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('source_folder', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('language', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('sha', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('repoevalconfig',
    sa.Column('repo_name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('sha', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.PrimaryKeyConstraint('sha')
    )
    op.create_table('test_modules',
    sa.Column('id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('filepath', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('targeted_files', sa.JSON(), nullable=False),
    sa.Column('test_metadata', sa.JSON(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('difftestgendataset',
    sa.Column('sha', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('patch', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('dataset_name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('source_files', sa.JSON(), nullable=False),
    sa.ForeignKeyConstraint(['sha'], ['commit.sha'], ),
    sa.PrimaryKeyConstraint('sha', 'dataset_name')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('difftestgendataset')
    op.drop_table('test_modules')
    op.drop_table('repoevalconfig')
    op.drop_table('repo_configs')
    op.drop_table('commit')
    # ### end Alembic commands ###
