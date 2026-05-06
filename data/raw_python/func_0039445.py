def _apply_default(self, config, default):
        """
        Apply default value to every str in config.
        Also ensure every TaggedValue has default in .tags
        """
        # No default, nothing to be done here
        if not default:
            return config

        # If the entire config is just a string, return default TaggedValue
        if isinstance(config, str):
            return TaggedValue(config, default, default, *self._global_tags)

        if isinstance(config, dict):
            # Recursively apply defaults for  each item in the config
            for key, val in config.items():
                config[key] = self._apply_default(val, default)

        elif isinstance(config, list):
            # Recursively apply defaults for each item in the config
            for i, val in enumerate(config):
                config[i] = self._apply_default(val, default)

        elif isinstance(config, TaggedValue):
            # Make sure each TaggedValue knows about the default tag
            config.tags.add(default)

        return config