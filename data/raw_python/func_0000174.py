def validate(ctx, sandbox):
    """Check if version of repository is semantic
    """
    m = RepoManager(ctx.obj['agile'])
    if not sandbox or m.can_release('sandbox'):
        click.echo(m.validate_version())