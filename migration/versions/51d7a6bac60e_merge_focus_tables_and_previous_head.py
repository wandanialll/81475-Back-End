"""Merge focus_tables and previous head

Revision ID: 51d7a6bac60e
Revises: 00157f408d07, focus_tables_20250617
Create Date: 2025-06-17 22:30:59.376449

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '51d7a6bac60e'
down_revision: Union[str, None] = ('00157f408d07', 'focus_tables_20250617')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
