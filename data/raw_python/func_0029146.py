def BLX(self, params):
        """
        BLX Rj

        Branch to the address in Rj, storing the next instruction in the Link Register
        """
        Rj = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(LR_or_general_purpose_registers=(Rj,))

        def BLX_func():
            self.register['LR'] = self.register['PC']  # No need for the + 1, PC already points to the next instruction
            self.register['PC'] = self.register[Rj]

        return BLX_func