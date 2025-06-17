"""Add focus_sessions and focus_labels tables"""
from alembic import op
import sqlalchemy as sa

revision = 'focus_tables_20250617'
down_revision = '5d59f3fa70c6'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'focus_sessions',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('attendance_id', sa.Integer, sa.ForeignKey('attendance.attendance_id', ondelete='CASCADE'), nullable=False),
        sa.Column('student_id', sa.String, nullable=True),
        sa.Column('focus_index', sa.Float, nullable=True),
        sa.Column('timestamp', sa.DateTime, server_default=sa.func.now()),
        sa.Column('closed', sa.Boolean, default=False)
    )
    op.create_table(
        'focus_labels',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('attendance_id', sa.Integer, sa.ForeignKey('attendance.attendance_id', ondelete='CASCADE'), nullable=False),
        sa.Column('student_id', sa.String, nullable=True),
        sa.Column('frame_id', sa.String, nullable=True),
        sa.Column('label', sa.String, nullable=False),
        sa.Column('timestamp', sa.DateTime, server_default=sa.func.now())
    )

def downgrade():
    op.drop_table('focus_labels')
    op.drop_table('focus_sessions')
