def check_immediate_unsigned_value(self, arg, bit):
        """
        Is the immediate within the unsigned value of 2**bit - 1

        Raises an exception if
        1. The immediate value is > 2**bit - 1
        :param arg: The parameter to check
        :param bit: The number of bits to use in 2**bit
        :return: The value of the immediate
        """
        i_num = self.check_immediate(arg)
        if (i_num > (2 ** bit - 1)) or (i_num < 0):
            raise iarm.exceptions.RuleError("Immediate {} is not an unsigned {} bit value".format(arg, bit))
        return i_num