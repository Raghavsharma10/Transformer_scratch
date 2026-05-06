def log(ctx, archive_name):
    '''
    Get the version log for an archive
    '''

    _generate_api(ctx)
    ctx.obj.api.get_archive(archive_name).log()