def history(ctx, archive_name):
    '''
    Get archive history
    '''

    _generate_api(ctx)
    var = ctx.obj.api.get_archive(archive_name)
    click.echo(pprint.pformat(var.get_history()))