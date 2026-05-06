def _keys_to_lower(self):
        """Convert key set to lowercase."""
        for k in list(self.keys()):
            val = super(CaseInsensitiveDict, self).__getitem__(k)
            super(CaseInsensitiveDict, self).__delitem__(k)
            self.__setitem__(CaseInsensitiveStr(k), val)