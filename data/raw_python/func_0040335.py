def eager_partial(__fn, *a, **kw):
        """Wrap a note for injection of an eagerly partially applied function.

        Use this instead of `partial` when eager injection is needed in place
        of lazy injection.
        """
        return (EAGER_PARTIAL, (__fn, a, tuple(kw.items())))