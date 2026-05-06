def _ondemand(f):
    """Decorator to only request information if not in cache already.
    """
    name = f.__name__

    def func(self, *args, **kwargs):
        if not args and not kwargs:
            if hasattr(self, '_%s' % name):
                return getattr(self, '_%s' % name)

            a = f(self, *args, **kwargs)
            setattr(self, '_%s' % name, a)
            return a
        else:
            return f(self, *args, **kwargs)
    func.__name__ = name
    return func