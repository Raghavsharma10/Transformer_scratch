def from_config(cls, name, config):
        """
        Behaves like the base Configurable class's `from_config()` except this
        makes sure that the `Pluggable` subclass with the given name is
        actually a properly installed plugin first.
        """
        installed_classes = cls.get_installed_classes()

        if name not in installed_classes:
            raise ValueError("Unknown/unavailable %s" % cls.__name__.lower())

        pluggable_class = installed_classes[name]

        pluggable_class.validate_config(config)

        instance = pluggable_class()
        if not instance.name:
            instance.name = name
        instance.apply_config(config)

        return instance