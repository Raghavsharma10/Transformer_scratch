def add_error(self, value, **kwargs):
        """Add an `Error` instance to this entry."""
        kwargs.update({ERROR.VALUE: value})
        self._add_cat_dict(Error, self._KEYS.ERRORS, **kwargs)
        return