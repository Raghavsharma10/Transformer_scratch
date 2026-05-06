def decode_return(codec="ascii"):
    """Decodes the return value of it isn't None"""

    def outer(f):
        def wrap(*args, **kwargs):
            res = f(*args, **kwargs)
            if res is not None:
                return res.decode(codec)
            return res
        return wrap
    return outer