def slice_naive(self, key):
        """
        Naively (on index) slice the field data and values.

        Args:
            key: Int, slice, or iterable to select data and values

        Returns:
            field: Sliced field object
        """
        cls = self.__class__
        key = check_key(self, key)
        enum = pd.Series(range(len(self)))
        enum.index = self.index
        values = self.field_values[enum[key].values]
        data = self.loc[key]
        return cls(data, field_values=values)