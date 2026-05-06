def configure(cls, name, config, prefix='depot.'):
        """Configures an application depot.

        This configures the application wide depot from a settings dictionary.
        The settings dictionary is usually loaded from an application configuration
        file where all the depot options are specified with a given ``prefix``.

        The default ``prefix`` is *depot.*, the minimum required setting
        is ``depot.backend`` which specified the required backend for files storage.
        Additional options depend on the choosen backend.

        """
        if name in cls._depots:
            raise RuntimeError('Depot %s has already been configured' % (name,))

        if cls._default_depot is None:
            cls._default_depot = name

        cls._depots[name] = cls.from_config(config, prefix)
        return cls._depots[name]