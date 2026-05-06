def is_bound(method):
    """
    Decorator that asserts the model instance is bound.

    Requires:
    1. an ``id`` attribute
    2. a ``url`` attribute
    2. a manager set
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self._is_bound:
            raise ValueError("%r must be bound to call %s()" % (self, method.__name__))
        return method(self, *args, **kwargs)
    return wrapper