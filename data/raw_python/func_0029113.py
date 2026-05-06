def validate(self, error_message=None):
        """
        :raises TypeError:
            If the value is not matched the type that the class represented.
        """

        if self.is_type():
            return

        if not error_message:
            error_message = "invalid value type"

        raise TypeError(
            "{}: expected={}, actual={}".format(error_message, self.typename, type(self._data))
        )