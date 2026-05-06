def add_nest(self, name=None, **kw):
        """A simple decorator which wraps :meth:`nestly.core.Nest.add`."""
        def deco(func):
            self.add(name or func.__name__, func, **kw)
            return func
        return deco