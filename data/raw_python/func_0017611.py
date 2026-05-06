def load_config_file(self, location):
        """
        Load a rotation scheme and other options from a configuration file.

        :param location: Any value accepted by :func:`coerce_location()`.
        :returns: The configured or given :class:`Location` object.
        """
        location = coerce_location(location)
        for configured_location, rotation_scheme, options in load_config_file(self.config_file, expand=False):
            if configured_location.match(location):
                logger.verbose("Loading configuration for %s ..", location)
                if rotation_scheme:
                    self.rotation_scheme = rotation_scheme
                for name, value in options.items():
                    if value:
                        setattr(self, name, value)
                # Create a new Location object based on the directory of the
                # given location and the execution context of the configured
                # location, because:
                #
                # 1. The directory of the configured location may be a filename
                #    pattern whereas we are interested in the expanded name.
                #
                # 2. The execution context of the given location may lack some
                #    details of the configured location.
                return Location(
                    context=configured_location.context,
                    directory=location.directory,
                )
        logger.verbose("No configuration found for %s.", location)
        return location