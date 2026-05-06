def convert(self):
        """
        :return: Converted value.
        :raises typepy.TypeConversionError:
            If the value cannot convert.
        """

        if self.is_type():
            return self.force_convert()

        raise TypeConversionError(
            "failed to convert from {} to {}".format(type(self._data).__name__, self.typename)
        )