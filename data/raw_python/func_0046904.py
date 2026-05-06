def convert_to_int(value, from_base):
        """
        Convert value to an int.

        :param value: the value to convert
        :type value: sequence of int
        :param int from_base: base of value
        :returns: the conversion result
        :rtype: int
        :raises ConvertError: if from_base is less than 2
        :raises ConvertError: if elements in value outside bounds

        Preconditions:
          * all integers in value must be at least 0
          * all integers in value must be less than from_base
          * from_base must be at least 2

        Complexity: O(len(value))
        """
        if from_base < 2:
            raise BasesValueError(
               from_base,
               "from_base",
               "must be greater than 2"
            )

        if any(x < 0 or x >= from_base for x in value):
            raise BasesValueError(
               value,
               "value",
               "elements must be at least 0 and less than %s" % from_base
            )
        return reduce(lambda x, y: x * from_base + y, value, 0)