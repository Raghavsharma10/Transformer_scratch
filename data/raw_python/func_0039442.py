def load(self, content):
        """Parse yaml content."""
        # Try parsing the YAML with global tags
        try:
            config = yaml.load(content, Loader=self._loader(self._global_tags))
        except yaml.YAMLError:
            raise InvalidConfigError(_("Config is not valid yaml."))

        # Try extracting just the tool portion
        try:
            config = config[self.tool]
        except (TypeError, KeyError):
            return None

        # If no scopes, just apply global default
        if not isinstance(config, dict):
            config = self._apply_default(config, self._global_default)
        else:
            # Figure out what scopes exist
            scoped_keys = set(key for key in self._scopes)

            # For every scope
            for key in config:
                # If scope has custom tags, apply
                if key in scoped_keys:
                    # local tags, and local default
                    tags, default = self._scopes[key]

                    # Inherit global default if no local default
                    if not default:
                        default = self._global_default

                    config[key] = self._apply_default(config[key], default)
                    self._apply_scope(config[key], tags)
                # Otherwise just apply global default
                else:
                    config[key] = self._apply_default(config[key], self._global_default)

        self._validate(config)

        return config