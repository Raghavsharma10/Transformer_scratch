def serializable_dict(d):
    """
    Return a dict like d, but with any un-json-serializable elements removed.
    """
    newd = {}
    for k in d.keys():
        if isinstance(d[k], type({})):
            newd[k] = serializable_dict(d[k])
            continue
        try:
            json.dumps({'k': d[k]})
            newd[k] = d[k]
        except:
            pass  # unserializable
    return newd