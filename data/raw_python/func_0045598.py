def copy(self, *args, **kwargs):
        """
        Make a copy of this object.

        Note:
            Copies both field data and field values.

        See Also:
            For arguments and description of behavior see `pandas docs`_.

        .. _pandas docs: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.copy.html
        """
        cls = self.__class__    # Note that type conversion does not perform copy
        data = pd.DataFrame(self).copy(*args, **kwargs)
        values = [field.copy() for field in self.field_values]
        return cls(data, field_values=values)