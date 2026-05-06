def adopt(self, payload, *args, flavour: ModuleType, **kwargs):
        """
        Concurrently run ``payload`` in the background

        If ``*args*`` and/or ``**kwargs`` are provided, pass them to ``payload`` upon execution.
        """
        if args or kwargs:
            payload = functools.partial(payload, *args, **kwargs)
        self._meta_runner.register_payload(payload, flavour=flavour)