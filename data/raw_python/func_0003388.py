def get(self, key, defaultvalue = None):
        """
        Support dict-like get (return a default value if not found)
        """
        (t, k) = self._getsubitem(key, False)
        if t is None:
            return defaultvalue
        else:
            return t.__dict__.get(k, defaultvalue)