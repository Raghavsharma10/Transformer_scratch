def get_tags(ctx, archive_name):
    '''
    Print tags assigned to an archive
    '''

    _generate_api(ctx)

    var = ctx.obj.api.get_archive(archive_name)

    click.echo(' '.join(var.get_tags()), nl=False)
    print('')