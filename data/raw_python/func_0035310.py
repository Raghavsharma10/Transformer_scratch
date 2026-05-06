def get_middleware(exclude=(), append=(),
                   current={'middleware': MIDDLEWARE_CLASSES}):
    """
    Returns MIDDLEWARE_CLASSES without the middlewares listed in exclude and
    with the middlewares listed in append.

    The use of a mutable dict is intentional, in order to preserve the state of
    the MIDDLEWARE_CLASSES tuple across multiple settings files.
    """

    current['middleware'] = tuple(
        [m for m in current['middleware'] if m not in exclude]
    ) + tuple(append)
    return current['middleware']