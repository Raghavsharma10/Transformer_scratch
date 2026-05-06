def search(ctx, tags, prefix=None):
    '''
    List all archives matching tag search criteria
    '''

    _generate_api(ctx)

    for i, match in enumerate(ctx.obj.api.search(*tags, prefix=prefix)):

        click.echo(match, nl=False)
        print('')