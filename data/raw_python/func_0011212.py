def getlist(self, key):
        """Returns a list of all the values for the named field. Returns an
        empty list if the key doesn't exist."""
        try:
            vals = _dict_getitem(self, key.lower())
        except KeyError:
            return []
        else:
            if isinstance(vals, tuple):
                return [vals[1]]
            else:
                return vals[1:]