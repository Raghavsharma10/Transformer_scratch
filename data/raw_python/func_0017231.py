def init():
    """Create database."""
    click.secho('Creating database {0}'.format(_db.engine.url),
                fg='green')
    if not database_exists(str(_db.engine.url)):
        create_database(str(_db.engine.url))