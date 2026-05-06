def UXTH(self, params):
        """
        UTXH Ra, Rb

        Zero extend the half word in Rb and store the result in Ra
        """
        Ra, Rb = self.get_two_parameters(r'\s*([^\s,]*),\s*([^\s,]*)(,\s*[^\s,]*)*\s*', params)

        self.check_arguments(low_registers=(Ra, Rb))

        def UXTH_func():
            self.register[Ra] = (self.register[Rb] & 0xFFFF)

        return UXTH_func