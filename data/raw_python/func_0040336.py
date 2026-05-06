def eager_partial_regardless(__fn, *a, **kw):
        """Wrap a note for injection of an eagerly partially applied function, or don't.

        Use this instead of `eager_partial partial` when binding a callable
        that may or may not have annotations.
        """
        return (EAGER_PARTIAL_REGARDLESS, (__fn, a, tuple(kw.items())))