def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    app_conf = dci_config.generate_conf()
    url = app_conf['SQLALCHEMY_DATABASE_URI']
    context.configure(
        url=url, target_metadata=target_metadata, literal_binds=True,
    )

    with context.begin_transaction():
        context.run_migrations()