def create(name, dry_run, verbose, query=None, parent=None):
    """Create new collection."""
    if parent is not None:
        parent = Collection.query.filter_by(name=parent).one().id
    collection = Collection(name=name, dbquery=query, parent_id=parent)
    db.session.add(collection)
    if verbose:
        click.secho('New collection: {0}'.format(collection))