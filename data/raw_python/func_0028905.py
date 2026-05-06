def LDRSH(self, params):
        """
        LDRSH Ra, [Rb, Rc]

        Load a half word from memory, sign extend, and put into Ra
        Ra, Rb, and Rc must be low registers
        """
        # TODO LDRSH cant use immediates
        Ra, Rb, Rc = self.get_three_parameters(self.THREE_PARAMETER_WITH_BRACKETS, params)

        self.check_arguments(low_registers=(Ra, Rb, Rc))

        def LDRSH_func():
            # TODO does memory read up?
            if (self.register[Rb] + self.register[Rc]) % 2 != 0:
                raise iarm.exceptions.HardFault(
                    "Memory access not half word aligned\nR{}: {}\nR{}: {}".format(Rb, self.register[Rb],
                                                                                   Rc, self.register[Rc]))
            self.register[Ra] = 0
            for i in range(2):
                self.register[Ra] |= (self.memory[self.register[Rb] + self.register[Rc] + i] << (8 * i))
            if self.register[Ra] & (1 << 15):
                self.register[Ra] |= (0xFFFF << 16)

        return LDRSH_func