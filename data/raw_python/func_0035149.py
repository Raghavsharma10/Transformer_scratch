def describe(db, zip, case_insensitive):
    """Show .dbf file statistics."""

    with open_db(db, zip, case_sensitive=not case_insensitive) as dbf:
        click.secho('Rows count: %s' % (dbf.prolog.records_count))
        click.secho('Fields:')
        for field in dbf.fields:
            click.secho('  %s: %s' % (field.type, field))