def tree(names):
    """Show the tree of the collection(s) specified."""
    # query
    query = Collection.query
    if names:
        query = query.filter(Collection.name.in_(names))
    else:
        query = query.filter(Collection.level == 1)
    # print tree
    tr = LeftAligned(traverse=AttributeTraversal())
    for coll in query.all():
        click.secho(tr(coll))