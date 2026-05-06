def check_register(self, arg):
        """
        Is the parameter a register in the form of 'R<d>',
        and if so is it within the bounds of registers defined

        Raises an exception if
        1. The parameter is not in the form of 'R<d>'
        2. <d> is outside the range of registers defined in the init value
            registers or _max_registers
        :param arg: The parameter to check
        :return: The number of the register
        """
        self.check_parameter(arg)
        match = re.search(self.REGISTER_REGEX, arg)
        if match is None:
            raise iarm.exceptions.RuleError("Parameter {} is not a register".format(arg))
        try:
            r_num = int(match.groups()[0])
        except ValueError:
            r_num = int(match.groups()[0], 16)
        except TypeError:
            if arg in 'lr|LR':
                return 14
            elif arg in 'sp|SP':
                return 13
            elif arg in 'fp|FP':
                return 7 ## TODO this could be 7 or 11 depending on THUMB and ARM mode http://www.keil.com/support/man/docs/armcc/armcc_chr1359124947957.htm
            else:
                raise
        if r_num > self._max_registers:
            raise iarm.exceptions.RuleError(
                "Register {} is greater than defined registers of {}".format(arg, self._max_registers))

        return r_num