def serializable(wrapped):
    """
    If a keyword argument 'serialize' with a True value is passed to the
    Wrapped function, the return of the wrapped function will be serialized.
    Nothing happens if the argument is not passed or the value is not True
    """

    @wraps(wrapped)
    def wrapper(*args, **kwargs):
        should_serialize = kwargs.pop('serialize', False)
        result = wrapped(*args, **kwargs)

        return serialize(result) if should_serialize else result

    if hasattr(wrapped, 'decorators'):
        wrapper.decorators = wrapped.decorators
        wrapper.decorators.append('serializable')
    else:
        wrapper.decorators = ['serializable']

    return wrapper