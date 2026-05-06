def upgrade():
    """Update database."""
    op.create_table(
        'transaction',
        sa.Column('issued_at', sa.DateTime(), nullable=True),
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('remote_addr', sa.String(length=50), nullable=True),
    )
    op.create_primary_key('pk_transaction', 'transaction', ['id'])
    if op._proxy.migration_context.dialect.supports_sequences:
        op.execute(CreateSequence(Sequence('transaction_id_seq')))