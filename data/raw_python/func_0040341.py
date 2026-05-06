def eager_partial(self, fn, *a, **kw):
        """Partially apply annotated callable, returning a partial function.

        By default, `partial` is lazy so that injections only happen when they
        are needed. Use `eager_partial` in place of `partial` when a guarantee
        of injection is needed at the time the partially applied function is
        created.

        `eager_partial` resolves arguments similarly to `partial` but relies on
        `functools.partial` for argument resolution when calling the final
        partial function.
        """
        args, kwargs = self.prepare_callable(fn, partial=True)
        args += a; kwargs.update(kw)
        return functools.partial(fn, *args, **kwargs)