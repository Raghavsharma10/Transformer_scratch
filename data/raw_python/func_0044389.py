def show(ctx):
    """
    Show migrations list
    """

    for app_name, app in ctx.obj['config']['apps'].items():
        click.echo(click.style(app_name, fg='green', bold=True))
        for migration in app['migrations']:
            applied = ctx.obj['db'].is_migration_applied(app_name, migration)
            click.echo('  {0} {1}'.format(migration, click.style('(applied)', bold=True) if applied else ''))