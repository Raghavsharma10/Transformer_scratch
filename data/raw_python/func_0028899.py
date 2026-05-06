def __validate_datetime_string(self):
        """
        This will require validating version string (such as "3.3.5").
        A version string could be converted to a datetime value if this
        validation is not executed.
        """

        try:
            try:
                StrictVersion(self._value)
                raise TypeConversionError(
                    "invalid datetime string: version string found {}".format(self._value)
                )
            except ValueError:
                pass
        except TypeError:
            raise TypeConversionError("invalid datetime string: type={}".format(type(self._value)))