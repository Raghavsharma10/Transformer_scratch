def add_source(self, allow_alias=False, **kwargs):
        """Add a `Source` instance to this entry."""
        if not allow_alias and SOURCE.ALIAS in kwargs:
            err_str = "`{}` passed in kwargs, this shouldn't happen!".format(
                SOURCE.ALIAS)
            self._log.error(err_str)
            raise RuntimeError(err_str)

        # Set alias number to be +1 of current number of sources
        if SOURCE.ALIAS not in kwargs:
            kwargs[SOURCE.ALIAS] = str(self.num_sources() + 1)
        source_obj = self._init_cat_dict(Source, self._KEYS.SOURCES, **kwargs)
        if source_obj is None:
            return None

        for item in self.get(self._KEYS.SOURCES, ''):
            if source_obj.is_duplicate_of(item):
                return item[item._KEYS.ALIAS]

        self.setdefault(self._KEYS.SOURCES, []).append(source_obj)
        return source_obj[source_obj._KEYS.ALIAS]