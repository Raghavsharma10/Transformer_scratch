def _build_cache():
    """Preprocess collection queries."""
    query = current_app.config['COLLECTIONS_DELETED_RECORDS']
    for collection in Collection.query.filter(
            Collection.dbquery.isnot(None)).all():
        yield collection.name, dict(
            query=query.format(dbquery=collection.dbquery),
            ancestors=set(_ancestors(collection)),
        )
    raise StopIteration