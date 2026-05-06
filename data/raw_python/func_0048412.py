def compose(first_func, second_func):
    """
    Compose two functions. Documentation is retrieved from the first one.

    Parameters
    ----------
    first_func
        The first, main, function.

    second_func
        The second, (less important) function.

    Returns
        function
        A new function.
    -------

    """

    @wraps(first_func)
    def composed_func(*args, **kwargs):
        return second_func((first_func(*args, **kwargs)))

    return composed_func