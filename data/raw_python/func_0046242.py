def _strip_trailing_zeros(value):
        """
        Strip trailing zeros from a list of ints.

        :param value: the value to be stripped
        :type value: list of str

        :returns: list with trailing zeros stripped
        :rtype: list of int
        """
        return list(
           reversed(
              list(itertools.dropwhile(lambda x: x == 0, reversed(value)))
           )
        )