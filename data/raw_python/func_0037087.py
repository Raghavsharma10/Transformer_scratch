def metadata(ctx, archive_name):
    '''
    Get an archive's metadata
    '''

    _generate_api(ctx)
    var = ctx.obj.api.get_archive(archive_name)
    click.echo(pprint.pformat(var.get_metadata()))