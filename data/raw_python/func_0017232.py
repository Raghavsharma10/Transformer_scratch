def destroy():
    """Drop database."""
    click.secho('Destroying database {0}'.format(_db.engine.url),
                fg='red', bold=True)
    if _db.engine.name == 'sqlite':
        try:
            drop_database(_db.engine.url)
        except FileNotFoundError as e:
            click.secho('Sqlite database has not been initialised',
                        fg='red', bold=True)
    else:
        drop_database(_db.engine.url)