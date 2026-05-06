def lint(ctx: click.Context, amend: bool = False, stage: bool = False):
    """
    Runs all linters

    Args:
        ctx: click context
        amend: whether or not to commit results
        stage: whether or not to stage changes
    """
    _lint(ctx, amend, stage)