def print_version(ctx: click.Context, _, value):
    """
    Prints current version then exits
    """
    if not value or ctx.resilient_parsing:
        return

    print(__version__)
    sys.exit(0)