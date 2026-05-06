def path(name):
    """Print path to root."""
    try:
        coll = Collection.query.filter(Collection.name == name).one()
        tr = LeftAligned(
            traverse=CollTraversalPathToRoot(coll.path_to_root().all()))
        click.echo(tr(coll))
    except NoResultFound:
        raise click.UsageError('Collection {0} not found'.format(name))