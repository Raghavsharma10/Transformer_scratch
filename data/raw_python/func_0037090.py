def filter_archives(ctx, prefix, pattern, engine):
    '''
    List all archives matching filter criteria
    '''

    _generate_api(ctx)

    # want to achieve behavior like click.echo(' '.join(matches))

    for i, match in enumerate(ctx.obj.api.filter(
            pattern, engine, prefix=prefix)):

        click.echo(match, nl=False)
        print('')