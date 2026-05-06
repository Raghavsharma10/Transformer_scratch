def _ancestors(collection):
    """Get the ancestors of the collection."""
    for index, c in enumerate(collection.path_to_root()):
        if index > 0 and c.dbquery is not None:
            raise StopIteration
        yield c.name
    raise StopIteration