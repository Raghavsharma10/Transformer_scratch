def load(f, dict_=dict):
    """Load and parse toml from a file object
    An additional argument `dict_` is used to specify the output type
    """
    if not f.read:
        raise ValueError('The first parameter needs to be a file object, ',
                         '%r is passed' % type(f))
    return loads(f.read(), dict_)