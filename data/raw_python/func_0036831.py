def from_config(cls, name, config):
        """
        Override of the base `from_config()` method that returns `None` if
        the name of the config file isn't "logging".

        We do this in case this `Configurable` subclass winds up sharing the
        root of the config directory with other subclasses.
        """
        if name != cls.name:
            return

        return super(Logging, cls).from_config(name, config)