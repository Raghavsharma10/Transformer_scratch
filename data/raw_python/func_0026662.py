def _init_cat_dict(self, cat_dict_class, key_in_self, **kwargs):
        """Initialize a CatDict object, checking for errors.
        """
        # Catch errors associated with crappy, but not unexpected data
        try:
            new_entry = cat_dict_class(self, key=key_in_self, **kwargs)
        except CatDictError as err:
            if err.warn:
                self._log.info("'{}' Not adding '{}': '{}'".format(self[
                    self._KEYS.NAME], key_in_self, str(err)))
            return None
        return new_entry