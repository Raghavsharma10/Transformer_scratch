def downgrade():
    """Downgrade database."""
    op.drop_table('transaction')
    if op._proxy.migration_context.dialect.supports_sequences:
        op.execute(DropSequence(Sequence('transaction_id_seq')))