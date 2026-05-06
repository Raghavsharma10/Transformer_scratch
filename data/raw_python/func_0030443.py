def walk_dict(d):
    """Walk a tree (nested dicts).

    For each 'path', or dict, in the tree, returns a 3-tuple containing:
    (path, sub-dicts, values)

    where:
    * path is the path to the dict
    * sub-dicts is a tuple of (key,dict) pairs for each sub-dict in this dict
    * values is a tuple of (key,value) pairs for each (non-dict) item in this dict

    """
    # nested dict keys
    nested_keys = tuple(k for k in list(d.keys()) if isinstance(d[k], dict))
    # key/value pairs for non-dicts
    items = tuple((k, d[k]) for k in list(d.keys()) if k not in nested_keys)

    # return path, key/sub-dict pairs, and key/value pairs
    yield ('/', [(k, d[k]) for k in nested_keys], items)

    # recurse each subdict
    for k in nested_keys:
        for res in walk_dict(d[k]):
            # for each result, stick key in path and pass on
            res = ('/%s' % k + res[0], res[1], res[2])
            yield res