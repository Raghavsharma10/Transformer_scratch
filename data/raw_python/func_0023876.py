def order_results_by(*fields):
    """A decorator that applies an ordering to the QuerySet returned by a
       function.
       """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kw):
            result = f(*args, **kw)
            return result.order_by(*fields)
        return wrapper
    return decorator