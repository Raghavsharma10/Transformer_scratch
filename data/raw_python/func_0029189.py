def set_APSR_flag_to_value(self, flag, value):
        """
        Set or clear flag in ASPR
        :param flag: The flag to set
        :param value: If value evaulates to true, it is set, cleared otherwise
        :return:
        """
        if flag == 'N':
            bit = 31
        elif flag == 'Z':
            bit = 30
        elif flag == 'C':
            bit = 29
        elif flag == 'V':
            bit = 28
        else:
            raise AttributeError("Flag {} does not exist in the APSR".format(flag))

        if value:
            self.register['APSR'] |= (1 << bit)
        else:
            self.register['APSR'] -= (1 << bit) if (self.register['APSR'] & (1 << bit)) else 0