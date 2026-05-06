def config_list(ctx):
    """ List the current configuration. """
    ingest_config_obj(ctx, silent=False)
    click.echo(json.dumps(ctx.obj['config'].to_dict(), indent=4))