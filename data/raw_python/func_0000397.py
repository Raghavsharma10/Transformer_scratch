def kong(ctx, namespace, yes):
    """Update the kong configuration
    """
    m = KongManager(ctx.obj['agile'], namespace=namespace)
    click.echo(utils.niceJson(m.create_kong(yes)))