def apply(ctx, name, verbose):
    """
    Apply migration
    """

    if name != 'all':  # specific migration
        try:
            app_name, target_migration = name.split('/', 2)
        except ValueError:
            raise click.ClickException("NAME format is <app>/<migration> or 'all'")

        apps = ctx.obj['config']['apps']
        if app_name not in apps.keys():
            raise click.ClickException('unknown app "{0}"'.format(app_name))

        app = apps[app_name]
        migrations = app['migrations']
        if target_migration not in migrations:
            raise click.ClickException('unknown migration "{0}"'.format(name))

        migrations = migrations[:migrations.index(target_migration) + 1]  # including all prevoius migrations
        for migration in migrations:
            click.echo(click.style('Applying {0}...'.format(click.style(migration, bold=True)), fg='blue'))

            if ctx.obj['db'].is_migration_applied(app_name, migration):
                click.echo(click.style('  SKIPPED.', fg='green'))
                continue

            try:
                snaql_factory = Snaql(app['path'], '')
                queries = snaql_factory.load_queries(migration + '.apply.sql').ordered_blocks

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

            ctx.obj['db'].fix_migration(app_name, migration)

    else:  # migrate everything
        for app_name, app in ctx.obj['config']['apps'].items():
            click.echo(click.style('Migrating {0}...'.format(click.style(app_name, bold=True)), fg='blue'))

            for migration in app['migrations']:
                click.echo('  Applying {0}...'.format(click.style(migration, bold=True)))

                if ctx.obj['db'].is_migration_applied(app_name, migration):
                    click.echo(click.style('    SKIPPED.', fg='green'))
                    continue

                try:
                    snaql_factory = Snaql(app['path'], '')
                    queries = snaql_factory.load_queries(migration + '.apply.sql').ordered_blocks

                    for query in queries:
                        if verbose:
                            click.echo('    ' + query())

                        ctx.obj['db'].query(query())

                except Exception as e:
                    click.echo(click.style('    FAILED.', fg='red'))
                    ctx.obj['db'].rollback()
                    raise click.ClickException('migration execution failed\n{0}'.format(e))

                click.echo(click.style('  OK.', fg='green'))

                ctx.obj['db'].commit()

                ctx.obj['db'].fix_migration(app_name, migration)