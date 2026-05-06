def clear(cls, fn):
        # type: (FunctionType) -> None
        """ Clear result cache on the given function.

        If the function has no cached result, this call will do nothing.

        Args:
            fn (FunctionType):
                The function whose cache should be cleared.
        """
        if hasattr(fn, cls.CACHE_VAR):
            delattr(fn, cls.CACHE_VAR)