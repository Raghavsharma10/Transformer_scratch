def _format_kwargs(func):
    """Decorator to handle formatting kwargs to the proper names expected by
    the associated function. The formats dictionary string keys will be used as
    expected function kwargs and the value list of strings will be renamed to
    the associated key string."""
    formats = {}
    formats['blk'] = ["blank"]
    formats['dft'] = ["default"]
    formats['hdr'] = ["header"]
    formats['hlp'] = ["help"]
    formats['msg'] = ["message"]
    formats['shw'] = ["show"]
    formats['vld'] = ["valid"]
    @wraps(func)
    def inner(*args, **kwargs):
        for k in formats.keys():
            for v in formats[k]:
                if v in kwargs:
                    kwargs[k] = kwargs[v]
                    kwargs.pop(v)
        return func(*args, **kwargs)
    return inner