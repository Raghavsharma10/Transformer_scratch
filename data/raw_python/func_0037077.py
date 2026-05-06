def create(
        ctx,
        archive_name,
        authority_name,
        versioned=True,
        tag=None,
        helper=False):
    '''
    Create an archive
    '''

    tags = list(tag)

    _generate_api(ctx)
    args, kwargs = _parse_args_and_kwargs(ctx.args)
    assert len(args) == 0, 'Unrecognized arguments: "{}"'.format(args)

    var = ctx.obj.api.create(
        archive_name,
        authority_name=authority_name,
        versioned=versioned,
        metadata=kwargs,
        tags=tags,
        helper=helper)

    verstring = 'versioned archive' if versioned else 'archive'
    click.echo('created {} {}'.format(verstring, var))