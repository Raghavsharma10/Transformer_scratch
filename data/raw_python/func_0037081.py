def get_dependencies(ctx, archive_name, version):
    '''
    List the dependencies of an archive
    '''

    _generate_api(ctx)

    var = ctx.obj.api.get_archive(archive_name)

    deps = []

    dependencies = var.get_dependencies(version=version)
    for arch, dep in dependencies.items():
        if dep is None:
            deps.append(arch)
        else:
            deps.append('{}=={}'.format(arch, dep))

    click.echo('\n'.join(deps))