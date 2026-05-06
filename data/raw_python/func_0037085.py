def cat(ctx, archive_name, version):
    '''
    Echo the contents of an archive
    '''

    _generate_api(ctx)
    var = ctx.obj.api.get_archive(archive_name)

    with var.open('r', version=version) as f:
        for chunk in iter(lambda: f.read(1024 * 1024), ''):
            click.echo(chunk)