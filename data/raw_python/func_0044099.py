def include(context, bundle_name, version):
    """Include a bundle of files into the internal space.

    Use bundle name if you simply want to inlcude the latest version.
    """
    store = Store(context.obj['database'], context.obj['root'])
    if version:
        version_obj = store.Version.get(version)
        if version_obj is None:
            click.echo(click.style('version not found', fg='red'))
    else:
        bundle_obj = store.bundle(bundle_name)
        if bundle_obj is None:
            click.echo(click.style('bundle not found', fg='red'))
        version_obj = bundle_obj.versions[0]

    try:
        include_version(context.obj['root'], version_obj)
    except VersionIncludedError as error:
        click.echo(click.style(error.message, fg='red'))
        context.abort()

    version_obj.included_at = dt.datetime.now()
    store.commit()
    click.echo(click.style('included all files!', fg='green'))