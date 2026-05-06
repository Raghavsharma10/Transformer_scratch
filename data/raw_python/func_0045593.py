def slice_cardinal(self, key):
        """
        Get the slice of this object by the value or values of the cardinal
        dimension.
        """
        cls = self.__class__
        key = check_key(self, key, cardinal=True)
        return cls(self[self[self._cardinal[0]].isin(key)])