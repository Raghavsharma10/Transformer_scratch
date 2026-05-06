def reflect(source, model, cache=None):
    '''Finds an object of class `model` with the same identifier as the
    `source` object'''
    if source is None:
        return None
    if cache and source in cache:
        return cache[source]
    db = object_session(source)
    ident = identity_key(instance=source)[1]
    assert ident is not None
    return db.query(model).get(ident)