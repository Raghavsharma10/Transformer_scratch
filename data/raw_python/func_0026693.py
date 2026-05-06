def sort_func(self, key):
        """Logic for sorting keys in a `Spectrum` relative to one another."""
        if key == self._KEYS.TIME:
            return 'aaa'
        if key == self._KEYS.DATA:
            return 'zzy'
        if key == self._KEYS.SOURCE:
            return 'zzz'
        return key