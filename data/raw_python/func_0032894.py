def reverse(func):
    """Given a function f(a, b), returns f(b, a)"""
    @wraps(func)
    def reversed_func(a, b):
        return func(b, a)
    reversed_func.__doc__ = "Reversed argument form of %s" % func.__doc__
    reversed_func.__name__ = "reversed %s" % func.__name__
    return reversed_func