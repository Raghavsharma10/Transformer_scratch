def Route(resource=None, methods=["get", "post", "put", "delete"],
          schema=None):
    """
    route
    """
    def _route(func):
        def wrapper(self, *args, **kwargs):
            # "test" argument means no wrap func this time,
            # return original func immediately.
            if kwargs.get("test", False):
                kwargs.pop("test")
                func(self, *args, **kwargs)

            _methods = methods
            if isinstance(methods, str):
                _methods = [methods]
            route = self.router.route(resource)
            for method in _methods:
                getattr(route, method)(func, schema)
        # Ordered by declare sequence
        # http://stackoverflow.com/questions/4459531/how-to-read-class-attributes-in-the-same-order-as-declared
        f_locals = sys._getframe(1).f_locals
        _order = len([v for v in f_locals.itervalues()
                     if hasattr(v, '__call__') and
                     hasattr(v, '__name__') and
                     v.__name__ == "wrapper"])
        wrapper.__dict__["_order"] = _order
        return wrapper
    return _route