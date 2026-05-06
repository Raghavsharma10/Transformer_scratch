def STRH(self, params):
        """
        STRH Ra, [Rb, Rc]
        STRH Ra, [Rb, #imm6_2]

        Store Ra into memory as a half word
        Ra, Rb, and Rc must be low registers
        """
        Ra, Rb, Rc = self.get_three_parameters(self.THREE_PARAMETER_WITH_BRACKETS, params)

        if self.is_immediate(Rc):
            self.check_arguments(low_registers=(Ra, Rb), imm5=(Rc,))

            def STRH_func():
                for i in range(2):
                    self.memory[self.register[Rb] + self.convert_to_integer(Rc[1:]) + i] = ((self.register[Ra] >> (8 * i)) & 0xFF)
        else:
            self.check_arguments(low_registers=(Ra, Rb, Rc))

            def STRH_func():
                for i in range(2):
                    self.memory[self.register[Rb] + self.register[Rc] + i] = ((self.register[Ra] >> (8 * i)) & 0xFF)

        return STRH_func