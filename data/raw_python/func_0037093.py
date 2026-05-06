def delete(ctx, archive_name):
    '''
    Delete an archive
    '''

    _generate_api(ctx)
    var = ctx.obj.api.get_archive(archive_name)

    var.delete()
    click.echo('deleted archive {}'.format(var))