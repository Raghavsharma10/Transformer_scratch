def partial_regardless(__fn, *a, **kw):
        """Wrap a note for injection of a partially applied function, or don't.

        Use this instead of `partial` when binding a callable that may or may
        not have annotations.
        """
        return (PARTIAL_REGARDLESS, (__fn, a, tuple(kw.items())))