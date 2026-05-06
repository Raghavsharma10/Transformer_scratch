def flip(f):
    """
    Calls the function f by flipping the first two positional
    arguments
    """

    def wrapped(*args, **kwargs):
        return f(*flip_first_two(args), **kwargs)

    f_spec = make_func_curry_spec(f)

    return curry_by_spec(f_spec, wrapped)