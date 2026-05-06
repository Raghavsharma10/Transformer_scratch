def LDRSB(self, params):
        """
        LDRSB Ra, [Rb, Rc]

        Load a byte from memory, sign extend, and put into Ra
        Ra, Rb, and Rc must be low registers
        """
        # TODO LDRSB cant use immediates
        Ra, Rb, Rc = self.get_three_parameters(self.THREE_PARAMETER_WITH_BRACKETS, params)

        self.check_arguments(low_registers=(Ra, Rb, Rc))

        def LDRSB_func():
            # TODO does memory read up?
            self.register[Ra] = 0
            self.register[Ra] |= self.memory[self.register[Rb] + self.register[Rc]]
            if self.register[Ra] & (1 << 7):
                self.register[Ra] |= (0xFFFFFF << 8)

        return LDRSB_func