def pretty_ref(obj: Any) -> str:
    """Pretty object reference using ``module.path:qual.name`` format"""
    try:
        return obj.__module__ + ':' + obj.__qualname__
    except AttributeError:
        return pretty_ref(type(obj)) + '(...)'