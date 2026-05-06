def replicate_no_merge(source, model, cache=None):
    '''Replicates the `source` object to `model` class and returns its
    reflection.'''
    # `cache` is used to break circular dependency: we need to replicate
    # attributes before merging target into the session, but replication of
    # some attributes may require target to be in session to avoid infinite
    # loop.
    if source is None:
        return None
    if cache is None:
        cache = {}
    elif source in cache:
        return cache[source]
    db = object_session(source)
    cls, ident = identity_key(instance=source)
    target = db.query(model).get(ident)
    if target is None:
        target = model()
    cache[source] = target
    try:
        replicate_attributes(source, target, cache=cache)
    except _PrimaryKeyIsNull:
        return None
    else:
        return target