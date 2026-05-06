def check(self, func=None, name=None):
        """
        A decorator to register a new Dockerflow check to be run
        when the /__heartbeat__ endpoint is called., e.g.::

            from dockerflow.flask import checks

            @dockerflow.check
            def storage_reachable():
                try:
                    acme.storage.ping()
                except SlowConnectionException as exc:
                    return [checks.Warning(exc.msg, id='acme.health.0002')]
                except StorageException as exc:
                    return [checks.Error(exc.msg, id='acme.health.0001')]

        or using a custom name::

            @dockerflow.check(name='acme-storage-check)
            def storage_reachable():
                # ...

        """
        if func is None:
            return functools.partial(self.check, name=name)

        if name is None:
            name = func.__name__

        self.logger.info('Registered Dockerflow check %s', name)

        @functools.wraps(func)
        def decorated_function(*args, **kwargs):
            self.logger.info('Called Dockerflow check %s', name)
            return func(*args, **kwargs)

        self.checks[name] = decorated_function
        return decorated_function