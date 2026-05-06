def start(ctx, debug, version, config):
    """Commands for devops operations"""
    ctx.obj = {}
    ctx.DEBUG = debug
    if os.path.isfile(config):
        with open(config) as fp:
            agile = json.load(fp)
    else:
        agile = {}
    ctx.obj['agile'] = agile
    if version:
        click.echo(__version__)
        ctx.exit(0)
    if not ctx.invoked_subcommand:
        click.echo(ctx.get_help())