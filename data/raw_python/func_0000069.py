def execute(self, payload, *args, flavour: ModuleType, **kwargs):
        """
        Synchronously run ``payload`` and provide its output

        If ``*args*`` and/or ``**kwargs`` are provided, pass them to ``payload`` upon execution.
        """
        if args or kwargs:
            payload = functools.partial(payload, *args, **kwargs)
        return self._meta_runner.run_payload(payload, flavour=flavour)