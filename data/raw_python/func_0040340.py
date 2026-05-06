def partial(self, fn, *user_args, **user_kwargs):
        """Return function with closure to lazily inject annotated callable.

        Repeat calls to the resulting function will reuse injections from the
        first call.

        Positional arguments are provided in this order:

        1. positional arguments provided by injector
        2. positional arguments provided in `partial_fn = partial(fn, *args)`
        3. positional arguments provided in `partial_fn(*args)`

        Keyword arguments are resolved in this order (later override earlier):

        1. keyword arguments provided by injector
        2. keyword arguments provided in `partial_fn = partial(fn, **kwargs)`
        3. keyword arguments provided in `partial_fn(**kargs)`

        Note that Python function annotations (in Python 3) are injected as
        keyword arguments, as documented in `annotate`, which affects the
        argument order here.

        `annotate.partial` accepts arguments in same manner as this `partial`.
        """
        self.get_annotations(fn) # Assert has annotations.
        def lazy_injection_fn(*run_args, **run_kwargs):
            arg_pack = getattr(lazy_injection_fn, 'arg_pack', None)
            if arg_pack is not None:
                pack_args, pack_kwargs = arg_pack
            else:
                jeni_args, jeni_kwargs = self.prepare_callable(fn, partial=True)
                pack_args = jeni_args + user_args
                pack_kwargs = {}
                pack_kwargs.update(jeni_kwargs)
                pack_kwargs.update(user_kwargs)
                lazy_injection_fn.arg_pack = (pack_args, pack_kwargs)
            final_args = pack_args + run_args
            final_kwargs = {}
            final_kwargs.update(pack_kwargs)
            final_kwargs.update(run_kwargs)
            return fn(*final_args, **final_kwargs)
        return lazy_injection_fn