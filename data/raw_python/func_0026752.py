def sort_func(self, key):
        """Used to sort keys when writing Entry to JSON format.

        Should be supplemented/overridden by inheriting classes.
        """
        if key == self._KEYS.SCHEMA:
            return 'aaa'
        if key == self._KEYS.NAME:
            return 'aab'
        if key == self._KEYS.SOURCES:
            return 'aac'
        if key == self._KEYS.ALIAS:
            return 'aad'
        if key == self._KEYS.MODELS:
            return 'aae'
        if key == self._KEYS.PHOTOMETRY:
            return 'zzy'
        if key == self._KEYS.SPECTRA:
            return 'zzz'
        return key