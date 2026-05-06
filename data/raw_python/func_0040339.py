def apply(self, fn, *a, **kw):
        """Fully apply annotated callable, returning callable's result."""
        args, kwargs = self.prepare_callable(fn)
        args += a; kwargs.update(kw)
        return fn(*args, **kwargs)