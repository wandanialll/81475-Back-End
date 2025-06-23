"""add focus tables

Revision ID: 1f075bb71796
Revises: 01fc04bc002b
Create Date: 2025-06-20 21:46:13.015034

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1f075bb71796'
down_revision: Union[str, None] = '01fc04bc002b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
