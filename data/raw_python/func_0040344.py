def eager_partial_regardless(self, fn, *a, **kw):
        """Like `eager_partial`, but applies if callable is not annotated."""
        if self.has_annotations(fn):
            return self.eager_partial(fn, *a, **kw)
        return functools.partial(fn, *a, **kw)