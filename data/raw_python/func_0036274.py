def get_installed_classes(cls):
        """
        Iterates over installed plugins associated with the `entry_point` and
        returns a dictionary of viable ones keyed off of their names.

        A viable installed plugin is one that is both loadable *and* a subclass
        of the Pluggable subclass in question.
        """
        installed_classes = {}
        for entry_point in pkg_resources.iter_entry_points(cls.entry_point):
            try:
                plugin = entry_point.load()
            except ImportError as e:
                logger.error(
                    "Could not load plugin %s: %s", entry_point.name, str(e)
                )
                continue

            if not issubclass(plugin, cls):
                logger.error(
                    "Could not load plugin %s:" +
                    " %s class is not subclass of %s",
                    entry_point.name, plugin.__class__.__name__, cls.__name__
                )
                continue

            if not plugin.validate_dependencies():
                logger.error(
                    "Could not load plugin %s:" +
                    " %s class dependencies not met",
                    entry_point.name, plugin.__name__
                )
                continue

            installed_classes[entry_point.name] = plugin

        return installed_classes