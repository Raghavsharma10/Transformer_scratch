def replicate(source, model, cache=None):
    '''Replicates the `source` object to `model` class and returns its
    reflection.'''
    target = replicate_no_merge(source, model, cache=cache)
    if target is not None:
        db = object_session(source)
        target = db.merge(target)
    return target