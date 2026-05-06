def update(
        ctx,
        archive_name,
        bumpversion='patch',
        prerelease=None,
        dependency=None,
        message=None,
        string=False,
        file=None):
    '''
    Update an archive with new contents
    '''

    _generate_api(ctx)

    args, kwargs = _parse_args_and_kwargs(ctx.args)
    assert len(args) == 0, 'Unrecognized arguments: "{}"'.format(args)

    dependencies_dict = _parse_dependencies(dependency)

    var = ctx.obj.api.get_archive(archive_name)
    latest_version = var.get_latest_version()

    if string:

        with var.open(
                'w+',
                bumpversion=bumpversion,
                prerelease=prerelease,
                dependencies=dependencies_dict,
                metadata=kwargs,
                message=message) as f:

            if file is None:
                for line in sys.stdin:
                    f.write(u(line))
            else:
                f.write(u(file))

    else:
        if file is None:
            file = click.prompt('enter filepath')

        var.update(
            file,
            bumpversion=bumpversion,
            prerelease=prerelease,
            dependencies=dependencies_dict,
            metadata=kwargs,
            message=message)

    new_version = var.get_latest_version()

    if latest_version is None and new_version is not None:
        bumpmsg = ' new version {} created.'.format(
            new_version)

    elif new_version != latest_version:
        bumpmsg = ' version bumped {} --> {}.'.format(
            latest_version, new_version)

    elif var.versioned:
        bumpmsg = ' version remains {}.'.format(latest_version)
    else:
        bumpmsg = ''

    click.echo('uploaded data to {}.{}'.format(var, bumpmsg))