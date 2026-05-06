def requires_libsodium(func):
    """
    Mark a function as requiring libsodium.

    If no libsodium support is detected, a `RuntimeError` is thrown.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        libsodium_check()
        return func(*args, **kwargs)

    return wrapper