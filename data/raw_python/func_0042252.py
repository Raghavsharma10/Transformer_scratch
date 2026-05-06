def upgrade():
    """Upgrade database."""
    op.create_table(
        'collection',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('dbquery', sa.Text(), nullable=True),
        sa.Column('rgt', sa.Integer(), nullable=False),
        sa.Column('lft', sa.Integer(), nullable=False),
        sa.Column('level', sa.Integer(), nullable=False),
        sa.Column('parent_id', sa.Integer(), nullable=True),
        sa.Column('tree_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ['parent_id'], ['collection.id'], ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        'collection_level_idx', 'collection', ['level'], unique=False
    )
    op.create_index('collection_lft_idx', 'collection', ['lft'], unique=False)
    op.create_index('collection_rgt_idx', 'collection', ['rgt'], unique=False)
    op.create_index(
        op.f('ix_collection_name'), 'collection', ['name'], unique=True
    )