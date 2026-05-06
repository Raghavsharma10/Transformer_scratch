def show(db, encoding, no_limit, zip, case_insensitive):
    """Show .dbf file contents (rows)."""

    limit = 15

    if no_limit:
        limit = float('inf')

    with open_db(db, zip, encoding=encoding, case_sensitive=not case_insensitive) as dbf:
        for idx, row in enumerate(dbf, 1):
            click.secho('')

            for key, val in row._asdict().items():
                click.secho('  %s: %s' % (key, val))

            if idx == limit:
                click.secho(
                    'Note: Output is limited to %s rows. Use --no-limit option to bypass.' % limit, fg='red')
                break