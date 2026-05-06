def _parse_default(self):
        """Parse the `Schema` for the `Bison` instance to create the set of
        default values.

        If no defaults are specified in the `Schema`, the default dictionary
        will not contain anything.
        """
        # the configuration changes, so we invalidate the cached config
        self._full_config = None

        if self.scheme:
            self._default.update(self.scheme.build_defaults())