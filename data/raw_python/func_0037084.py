def download(ctx, archive_name, filepath, version):
    '''
    Download an archive
    '''

    _generate_api(ctx)
    var = ctx.obj.api.get_archive(archive_name)

    if version is None:
        version = var.get_default_version()

    var.download(filepath, version=version)

    archstr = var.archive_name +\
        '' if (not var.versioned) else ' v{}'.format(version)

    click.echo('downloaded{} to {}'.format(archstr, filepath))