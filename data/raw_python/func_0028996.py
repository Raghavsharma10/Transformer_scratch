def MVNS(self, params):
        """
        MVNS Ra, Rb

        Negate the value in Rb and store it in Ra
        Ra and Rb must be a low register
        """
        Ra, Rb = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)

        self.check_arguments(low_registers=(Ra, Rb))

        def MVNS_func():
            self.register[Ra] = ~self.register[Rb]
            self.set_NZ_flags(self.register[Ra])

        return MVNS_func