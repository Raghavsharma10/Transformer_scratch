def REV16(self, params):
        """
        REV16 Ra, Rb

        Reverse the byte order of the half words in register Rb and store the result in Ra
        """
        Ra, Rb = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)

        self.check_arguments(low_registers=(Ra, Rb))

        def REV16_func():
            self.register[Ra] = ((self.register[Rb] & 0xFF00FF00) >> 8) | \
                                ((self.register[Rb] & 0x00FF00FF) << 8)

        return REV16_func