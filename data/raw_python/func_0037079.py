def update_metadata(ctx, archive_name):
    '''
    Update an archive's metadata
    '''

    _generate_api(ctx)
    args, kwargs = _parse_args_and_kwargs(ctx.args)
    assert len(args) == 0, 'Unrecognized arguments: "{}"'.format(args)

    var = ctx.obj.api.get_archive(archive_name)

    var.update_metadata(metadata=kwargs)