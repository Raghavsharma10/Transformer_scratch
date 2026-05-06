def set_dependencies(ctx, archive_name, dependency=None):
    '''
    Set the dependencies of an archive
    '''

    _generate_api(ctx)
    kwargs = _parse_dependencies(dependency)

    var = ctx.obj.api.get_archive(archive_name)

    var.set_dependencies(dependencies=kwargs)