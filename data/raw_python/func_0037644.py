def normalize_signature(func):
    """Decorator.  Combine args and kwargs. Unpack single item tuples."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs:
            args = args, kwargs

        if len(args) is 1:
            args = args[0]

        return func(args)

    return wrapper