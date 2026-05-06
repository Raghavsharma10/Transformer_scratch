def revert(ctx, name, verbose):
    """
    Revert migration
    """

    try:
        app_name, target_migration = name.split('/', 2)
    except ValueError:
        raise click.ClickException('NAME format is <app>/<migration>')

    apps = ctx.obj['config']['apps']
    if app_name not in apps.keys():
        raise click.ClickException('unknown app "{0}"'.format(app_name))

    app = apps[app_name]
    migrations = app['migrations']
    if target_migration not in migrations:
        raise click.ClickException('unknown migration "{0}"'.format(name))

    mig_idx = migrations.index(target_migration)
    migrations = migrations[-len(migrations) + mig_idx:]  # all migrations after target_migration
    migrations = migrations[::-1]  # in reversed order

    for migration in migrations:
        click.echo(
            click.style('Reverting {0}...'.format(click.style(app_name + '/' + migration, bold=True)), fg='blue'))

        if not ctx.obj['db'].is_migration_applied(app_name, migration):
            click.echo(click.style('  SKIPPED.', fg='green'))
            continue
        try:
            snaql_factory = Snaql(app['path'], '')
            queries = snaql_factory.load_queries(migration + '.revert.sql').ordered_blocks

            for query in queries:
                if verbose:
                    click.echo('    ' + query())

                ctx.obj['db'].query(query())

        except Exception as e:
            click.echo(click.style('  FAILED.', fg='red'))
            ctx.obj['db'].rollback()

            raise click.ClickException('migration execution failed\n{0}'.format(e))

        click.echo(click.style('  OK.', fg='green'))

        ctx.obj['db'].commit()

        ctx.obj['db'].revert_migration(app_name, migration)