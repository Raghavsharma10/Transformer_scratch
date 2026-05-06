def check_immediate_value(self, arg, _max, _min=0):
        """
        Is the immediate within the range of [_min, _max]

        Raises an exception if
        1. The immediate value is < _min or > _max
        :param arg: The parameter to check
        :param _max: The maximum value
        :param _min: The minimum value, optional, default is zero
        :return: The immediate value
        """
        i_num = self.check_immediate(arg)
        if (i_num > _max) or (i_num < _min):
            raise iarm.exceptions.RuleError("Immediate {} is not within the range of [{}, {}]".format(arg, _min, _max))
        return i_num