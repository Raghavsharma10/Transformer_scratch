def apply_regardless(self, fn, *a, **kw):
        """Like `apply`, but applies if callable is not annotated."""
        if self.has_annotations(fn):
            return self.apply(fn, *a, **kw)
        return fn(*a, **kw)