def convert_from_int(value, to_base):
        """
        Convert int value to a base.

        :param int value: the value to convert, must be at least 0
        :param int to_base: base of result, must be at least 2
        :returns: the conversion result
        :rtype: list of int
        :raises BasesValueError: if value is less than 0
        :raises BasesValueError: if to_base is less than 2

        Preconditions:
          * to_base must be at least 2

        Complexity: O(log_{to_base}(value))
        """
        if value < 0:
            raise BasesValueError(value, "value", "must be at least 0")

        if to_base < 2:
            raise BasesValueError(to_base, "to_base", "must be at least 2")

        result = []
        while value != 0:
            (value, rem) = divmod(value, to_base)
            result.append(rem)
        result.reverse()
        return result