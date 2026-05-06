def SXTB(self, params):
        """
        STXB Ra, Rb

        Sign extend the byte in Rb and store the result in Ra
        """
        Ra, Rb = self.get_two_parameters(r'\s*([^\s,]*),\s*([^\s,]*)(,\s*[^\s,]*)*\s*', params)

        self.check_arguments(low_registers=(Ra, Rb))

        def SXTB_func():
            if self.register[Rb] & (1 << 7):
                self.register[Ra] = 0xFFFFFF00 + (self.register[Rb] & 0xFF)
            else:
                self.register[Ra] = (self.register[Rb] & 0xFF)

        return SXTB_func