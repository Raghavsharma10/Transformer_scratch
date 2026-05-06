def add_tags(ctx, archive_name, tags):
    '''
    Add tags to an archive
    '''

    _generate_api(ctx)

    var = ctx.obj.api.get_archive(archive_name)

    var.add_tags(*tags)