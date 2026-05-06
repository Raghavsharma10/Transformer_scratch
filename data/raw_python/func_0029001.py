def SXTH(self, params):
        """
        STXH Ra, Rb

        Sign extend the half word in Rb and store the result in Ra
        """
        Ra, Rb = self.get_two_parameters(r'\s*([^\s,]*),\s*([^\s,]*)(,\s*[^\s,]*)*\s*', params)

        self.check_arguments(low_registers=(Ra, Rb))

        def SXTH_func():
            if self.register[Rb] & (1 << 15):
                self.register[Ra] = 0xFFFF0000 + (self.register[Rb] & 0xFFFF)
            else:
                self.register[Ra] = (self.register[Rb] & 0xFFFF)

        return SXTH_func