def to_imgur_format(params):
    """Convert the parameters to the format Imgur expects."""
    if params is None:
        return None
    return dict((k, convert_general(val)) for (k, val) in params.items())