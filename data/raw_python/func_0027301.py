def convert_result(converter):
    """Decorator that can convert the result of a function call."""

    def decorate(fn):
        @inspection.wraps(fn)
        def new_fn(*args, **kwargs):
            return converter(fn(*args, **kwargs))

        return new_fn

    return decorate