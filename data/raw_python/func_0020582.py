def encode(obj):
    """
    Encode one argument/object to json
    """
    if hasattr(obj, 'json'):
        return obj.json
    if hasattr(obj, '__json__'):
        return obj.__json__()
    return dumps(obj)