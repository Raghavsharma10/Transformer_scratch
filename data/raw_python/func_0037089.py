def versions(ctx, archive_name):
    '''
    Get an archive's versions
    '''

    _generate_api(ctx)

    var = ctx.obj.api.get_archive(archive_name)
    click.echo(pprint.pformat(map(str, var.get_versions())))