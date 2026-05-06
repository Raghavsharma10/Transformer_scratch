def query(name):
    """Print the collection query."""
    collection = Collection.query.filter_by(name=name).one()
    click.echo(collection.dbquery)