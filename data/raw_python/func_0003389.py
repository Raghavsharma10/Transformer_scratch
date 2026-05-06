def setdefault(self, key, defaultvalue = None):
        """
        Support dict-like setdefault (create if not existed)
        """
        (t, k) = self._getsubitem(key, True)
        return t.__dict__.setdefault(k, defaultvalue)