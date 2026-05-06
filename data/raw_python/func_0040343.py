def partial_regardless(self, fn, *a, **kw):
        """Like `partial`, but applies if callable is not annotated."""
        if self.has_annotations(fn):
            return self.partial(fn, *a, **kw)
        else:
            return functools.partial(fn, *a, **kw)