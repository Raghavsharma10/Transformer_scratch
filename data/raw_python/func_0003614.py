def from_config(cls, config, prefix='depot.'):
        """Creates a new depot from a settings dictionary.

        Behaves like the :meth:`configure` method but instead of configuring the application
        depot it creates a new one each time.
        """
        config = config or {}

        # Get preferred storage backend
        backend = config.get(prefix + 'backend', 'depot.io.local.LocalFileStorage')

        # Get all options
        prefixlen = len(prefix)
        options = dict((k[prefixlen:], config[k]) for k in config.keys() if k.startswith(prefix))

        # Backend is already passed as a positional argument
        options.pop('backend', None)
        return cls._new(backend, **options)