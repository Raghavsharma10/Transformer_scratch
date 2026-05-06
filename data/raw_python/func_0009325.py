def _is_string(cls, arg):
        """
        Return True if arg is a string value,
        and False if arg is a logical value (ANY or NA).

        :param string arg: string to check
        :returns: True if value is a string, False if it is a logical value.
        :rtype: boolean

        This function is a support function for _compare().
        """

        isAny = arg == CPEComponent2_3_WFN.VALUE_ANY
        isNa = arg == CPEComponent2_3_WFN.VALUE_NA

        return not (isAny or isNa)