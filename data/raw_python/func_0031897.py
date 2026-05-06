def add(self, value):
        """
        Add a value to the reservoir
        The value will be casted to a floating-point, so a TypeError or a
        ValueError may be raised.
        """

        if not isinstance(value, float):
            value = float(value)

        return self._do_add(value)