def collection_updated_percolator(mapper, connection, target):
    """Create percolator when collection is created.

    :param mapper: Not used. It keeps the function signature.
    :param connection: Not used. It keeps the function signature.
    :param target: Collection where the percolator should be updated.
    """
    delete_collection_percolator(target)
    if target.dbquery is not None:
        new_collection_percolator(target)