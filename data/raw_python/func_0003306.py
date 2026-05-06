def http(container = None):
    "wrap a WSGI-style class method to a HTTPRequest event handler"
    def decorator(func):
        @functools.wraps(func)
        def handler(self, event):
            return _handler(self if container is None else container, event, lambda env: func(self, env))
        return handler
    return decorator