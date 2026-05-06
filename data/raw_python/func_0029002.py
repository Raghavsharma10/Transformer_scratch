def UXTB(self, params):
        """
        UTXB Ra, Rb

        Zero extend the byte in Rb and store the result in Ra
        """
        Ra, Rb = self.get_two_parameters(r'\s*([^\s,]*),\s*([^\s,]*)(,\s*[^\s,]*)*\s*', params)

        self.check_arguments(low_registers=(Ra, Rb))

        def UXTB_func():
            self.register[Ra] = (self.register[Rb] & 0xFF)

        return UXTB_func