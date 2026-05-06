def slice_naive(self, key):
        """
        Slice a data object based on its index, either by value (.loc) or
        position (.iloc).

        Args:
            key: Single index value, slice, tuple, or list of indices/positionals

        Returns:
            data: Slice of self
        """
        cls = self.__class__
        key = check_key(self, key)
        return cls(self.loc[key])