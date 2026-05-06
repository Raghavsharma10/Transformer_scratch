def _check_cat_dict_source(self, cat_dict_class, key_in_self, **kwargs):
        """Check that a source exists and that a quantity isn't erroneous."""
        # Make sure that a source is given
        source = kwargs.get(cat_dict_class._KEYS.SOURCE, None)
        if source is None:
            raise CatDictError(
                "{}: `source` must be provided!".format(self[self._KEYS.NAME]),
                warn=True)
        # Check that source is a list of integers
        for x in source.split(','):
            if not is_integer(x):
                raise CatDictError(
                    "{}: `source` is comma-delimited list of "
                    " integers!".format(self[self._KEYS.NAME]),
                    warn=True)
        # If this source/data is erroneous, skip it
        if self.is_erroneous(key_in_self, source):
            self._log.info("This source is erroneous, skipping")
            return None
        # If this source/data is private, skip it
        if (self.catalog.args is not None and not self.catalog.args.private and
                self.is_private(key_in_self, source)):
            self._log.info("This source is private, skipping")
            return None
        return source