"""add face embedding table

Revision ID: 01fc04bc002b
Revises: fc25c17e0587
Create Date: 2025-06-20 19:36:10.563861

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import datetime


# revision identifiers, used by Alembic.
revision: str = '01fc04bc002b'
down_revision: Union[str, None] = 'fc25c17e0587'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'face_embeddings',
        sa.Column('embedding_id', sa.Integer, primary_key=True),
        sa.Column('student_id', sa.Integer, sa.ForeignKey('students.student_id'), nullable=False, unique=True),
        sa.Column('embedding', sa.PickleType, nullable=False),
        sa.Column('created_at', sa.DateTime, default=datetime.datetime.utcnow)
    )
    pass


def downgrade() -> None:
    op.drop_table('face_embeddings')
    pass
