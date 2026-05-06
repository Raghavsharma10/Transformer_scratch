def carry_in(value, carry, base):
        """
        Add a carry digit to a number represented by ``value``.

        :param value: the value
        :type value: list of int
        :param int carry: the carry digit (>= 0)
        :param int base: the base (>= 2)

        :returns: carry-out and result
        :rtype: tuple of int * (list of int)

	Complexity: O(len(value))
        """
        if base < 2:
            raise BasesValueError(base, "base", "must be at least 2")

        if any(x < 0 or x >= base for x in value):
            raise BasesValueError(
               value,
               "value",
               "elements must be at least 0 and less than %s" % base
            )

        if carry < 0 or carry >= base:
            raise BasesValueError(
               carry,
               "carry",
               "carry must be less than %s" % base
            )

        result = []
        for val in reversed(value):
            (carry, new_val) = divmod(val + carry, base)
            result.append(new_val)

        return (carry, list(reversed(result)))