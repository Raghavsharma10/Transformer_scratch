def statichttp(container = None):
    "wrap a WSGI-style function to a HTTPRequest event handler"
    def decorator(func):
        @functools.wraps(func)
        def handler(event):
            return _handler(container, event, func)
        if hasattr(func, '__self__'):
            handler.__self__ = func.__self__
        return handler
    return decorator