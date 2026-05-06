def from_config(cls, name, config):
        """
        Returns a Configurable instance with the given name and config.

        By default this is a simple matter of calling the constructor, but
        subclasses that are also `Pluggable` instances override this in order
        to check that the plugin is installed correctly first.
        """

        cls.validate_config(config)

        instance = cls()
        if not instance.name:
            instance.name = config.get("name", name)
        instance.apply_config(config)

        return instance