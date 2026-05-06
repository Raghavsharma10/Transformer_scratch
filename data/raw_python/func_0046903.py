def convert(cls, value, from_base, to_base):
        """
        Convert value from a base to a base.

        :param value: the value to convert
        :type value: sequence of int
        :param int from_base: base of value
        :param int to_base: base of result
        :returns: the conversion result
        :rtype: list of int
        :raises ConvertError: if from_base is less than 2
        :raises ConvertError: if to_base is less than 2
        :raises ConvertError: if elements in value outside bounds

        Preconditions:
          * all integers in value must be no less than 0
          * from_base, to_base must be at least 2

        Complexity: O(len(value))
        """
        return cls.convert_from_int(
           cls.convert_to_int(value, from_base),
           to_base
        )