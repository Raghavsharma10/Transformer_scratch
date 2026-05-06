def deepcp(data):
    """Use ujson to do deep_copy"""
    import ujson
    try:
        return ujson.loads(ujson.dumps(data))
    except Exception:
        return copy.deepcopy(data)