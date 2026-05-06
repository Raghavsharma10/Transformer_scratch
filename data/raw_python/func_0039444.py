def _validate(self, config):
        """Check whether every TaggedValue has a valid tag, otherwise raise InvalidConfigError"""
        if isinstance(config, dict):
            # Recursively validate each item in the config
            for val in config.values():
                self._validate(val)

        elif isinstance(config, list):
            # Recursively validate each item in the config
            for item in config:
                self._validate(item)

        elif isinstance(config, TaggedValue):
            tagged_value = config

            # if tagged_value is invalid, error
            if tagged_value.tag not in tagged_value.tags:
                raise InvalidConfigError(_("{} is not a valid tag for {}".format(tagged_value.tag, self.tool)))