def MULS(self, params):
        """
        MULS Ra, Rb, Ra

        Multiply Rb and Ra together and store the result in Ra.
        Set the NZ flags.
        Ra and Rb must be low registers
        The first and last operand must be the same register
        """
        Ra, Rb, Rc = self.get_three_parameters(self.THREE_PARAMETER_COMMA_SEPARATED, params)

        self.check_arguments(low_registers=(Ra, Rb, Rc))
        if Ra != Rc:
            raise iarm.exceptions.RuleError("Third parameter {} is not the same as the first parameter {}".format(Rc, Ra))

        # MULS Ra, Rb, Ra
        def MULS_func():
            self.register[Ra] = self.register[Rb] * self.register[Rc]
            self.set_NZ_flags(self.register[Ra])

        return MULS_func