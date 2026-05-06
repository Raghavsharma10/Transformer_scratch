def create(verbose):
    """Create tables."""
    click.secho('Creating all tables!', fg='yellow', bold=True)
    with click.progressbar(_db.metadata.sorted_tables) as bar:
        for table in bar:
            if verbose:
                click.echo(' Creating table {0}'.format(table))
            table.create(bind=_db.engine, checkfirst=True)
    create_alembic_version_table()
    click.secho('Created all tables!', fg='green')