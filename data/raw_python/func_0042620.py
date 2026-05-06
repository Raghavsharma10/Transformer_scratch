def delete(name, dry_run, verbose):
    """Delete a collection."""
    collection = Collection.query.filter_by(name=name).one()
    if verbose:
        tr = LeftAligned(traverse=AttributeTraversal())
        click.secho(tr(collection), fg='red')
    db.session.delete(collection)