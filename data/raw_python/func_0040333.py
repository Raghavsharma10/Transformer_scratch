def partial(__fn, *a, **kw):
        """Wrap a note for injection of a partially applied function.

        This allows for annotated functions to be injected for composition::

            from jeni import annotate

            @annotate('foo', bar=annotate.maybe('bar'))
            def foobar(foo, bar=None):
                return

            @annotate('foo', annotate.partial(foobar))
            def bazquux(foo, fn):
                # fn: injector.partial(foobar)
                return

        Keyword arguments are treated as `maybe` when using partial, in order
        to allow partial application of only the notes which can be provided,
        where the caller could then apply arguments known to be unavailable in
        the injector. Note that with Python 3 function annotations, all
        annotations are injected as keyword arguments.

        Injections on the partial function are lazy and not applied until the
        injected partial function is called. See `eager_partial` to inject
        eagerly.
        """
        return (PARTIAL, (__fn, a, tuple(kw.items())))