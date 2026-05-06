def packb(obj, **kwargs):
    """wrap msgpack.packb, setting use_bin_type=True by default"""
    kwargs.setdefault('use_bin_type', True)
    return msgpack.packb(obj, **kwargs)