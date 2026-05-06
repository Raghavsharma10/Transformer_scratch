def attach(names, parent, dry_run, verbose):
    """Attach collection(s) to a parent."""
    parent = Collection.query.filter_by(name=parent).one()
    collections = Collection.query.filter(Collection.name.in_(names)).all()
    for collection in collections:
        collection.move_inside(parent.id)
        if verbose:
            click.secho(
                'Collection "{0}" is being attached to "{1}".'.format(
                    collection.name, parent
                ), fg='green')