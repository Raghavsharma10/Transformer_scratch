def discrete(cats, name='discrete'):
    """Return a class category that shows the encoding"""
    import json
    ks = list(cats)
    for key in ks:
        if isinstance(key, bytes):
            cats[key.decode('utf-8')] = cats.pop(key)
    return 'discrete(' + json.dumps([cats, name]) + ')'