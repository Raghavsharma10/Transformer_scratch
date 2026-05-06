def jsonhash(obj, root=True, exclude=None, hash_func=_jsonhash_sha1):
    '''
    calculate the objects hash based on all field values
    '''
    if isinstance(obj, Mapping):
        # assumption: using in against set() is faster than in against list()
        if root and exclude:
            obj = {k: v for k, v in obj.iteritems() if k not in exclude}
        # frozenset's don't guarantee order; use sorted tuples
        # which means different python interpreters can return
        # back frozensets with different hash values even when
        # the content of the object is exactly the same
        result = sorted(
            (k, jsonhash(v, False)) for k, v in obj.iteritems())
    elif isinstance(obj, list):
        # FIXME: should lists be sorted for consistent hashes?
        # when the object is the same, just different list order?
        result = tuple(jsonhash(e, False) for e in obj)
    else:
        result = obj
    if root:
        result = unicode(hash_func(result))
    return result