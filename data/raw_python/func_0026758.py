def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    engine = create_engine(get_url())

    connection = engine.connect()
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table="alembic_ziggurat_foundations_version",
        transaction_per_migration=True,
    )

    try:
        with context.begin_transaction():
            context.run_migrations()
    finally:
        connection.close()