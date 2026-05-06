def REV(self, params):
        """
        REV Ra, Rb

        Reverse the byte order in register Rb and store the result in Ra
        """
        Ra, Rb = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)

        self.check_arguments(low_registers=(Ra, Rb))

        def REV_func():
            self.register[Ra] = ((self.register[Rb] & 0xFF000000) >> 24) | \
                                ((self.register[Rb] & 0x00FF0000) >> 8) | \
                                ((self.register[Rb] & 0x0000FF00) << 8) | \
                                ((self.register[Rb] & 0x000000FF) << 24)

        return REV_func