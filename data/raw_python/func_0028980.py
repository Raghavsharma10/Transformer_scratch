def CMN(self, params):
        """
        CMN Ra, Rb

        Add the two registers and set the NZCV flags
        The result is discarded
        Ra and Rb must be low registers
        """
        Ra, Rb = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)

        self.check_arguments(low_registers=(Ra, Rb))

        # CMN Ra, Rb
        def CMN_func():
            self.set_NZCV_flags(self.register[Ra], self.register[Rb],
                                self.register[Ra] + self.register[Rb], 'add')

        return CMN_func