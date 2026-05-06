def BX(self, params):
        """
        BX Rj

        Jump to the address in the Link Register
        """
        Rj = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(LR_or_general_purpose_registers=(Rj,))

        def BX_func():
            self.register['PC'] = self.register[Rj]

        return BX_func