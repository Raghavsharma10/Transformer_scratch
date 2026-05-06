def TST(self, params):
        """
        TST Ra, Rb

        AND Ra and Rb together and update the NZ flag. The result is not set
        The equivalent of `Ra & Rc`
        Ra and Rb must be low registers
        """
        Ra, Rb = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)

        self.check_arguments(low_registers=(Ra, Rb))

        def TST_func():
            result = self.register[Ra] & self.register[Rb]
            self.set_NZ_flags(result)

        return TST_func