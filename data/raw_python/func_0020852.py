def upgrade():
    """Upgrade database."""
    op.create_table(
        'oaiserver_set',
        sa.Column('created', sa.DateTime(), nullable=False),
        sa.Column('updated', sa.DateTime(), nullable=False),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('spec', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('search_pattern', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('spec')
    )
    op.create_index(
        op.f('ix_oaiserver_set_name'),
        'oaiserver_set',
        ['name'],
        unique=False
    )