def ingest_config_obj(ctx, *, silent=True):
    """ Ingest the configuration object into the click context. """
    try:
        ctx.obj['config'] = Config.from_file(ctx.obj['config_path'])
    except ConfigLoadError as err:
        click.echo(_style(ctx.obj['show_color'], str(err), fg='red', bold=True))
        if not silent:
            raise click.Abort()