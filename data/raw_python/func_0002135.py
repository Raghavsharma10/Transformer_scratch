def pop(self, key, *args, **kwargs):
        """Remove and return the value associated with case-insensitive ``key``."""
        return super(CaseInsensitiveDict, self).pop(CaseInsensitiveStr(key))