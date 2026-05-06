def drop(verbose):
    """Drop tables."""
    click.secho('Dropping all tables!', fg='red', bold=True)
    with click.progressbar(reversed(_db.metadata.sorted_tables)) as bar:
        for table in bar:
            if verbose:
                click.echo(' Dropping table {0}'.format(table))
            table.drop(bind=_db.engine, checkfirst=True)
        drop_alembic_version_table()
    click.secho('Dropped all tables!', fg='green')