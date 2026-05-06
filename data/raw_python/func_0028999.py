def REVSH(self, params):
        """
        REVSH

        Reverse the byte order in the lower half word in Rb and store the result in Ra.
        If the result of the result is signed, then sign extend
        """
        Ra, Rb = self.get_two_parameters(r'\s*([^\s,]*),\s*([^\s,]*)(,\s*[^\s,]*)*\s*', params)

        self.check_arguments(low_registers=(Ra, Rb))

        def REVSH_func():
            self.register[Ra] = ((self.register[Rb] & 0x0000FF00) >> 8) | \
                                ((self.register[Rb] & 0x000000FF) << 8)
            if self.register[Ra] & (1 << 15):
                self.register[Ra] |= 0xFFFF0000

        return REVSH_func