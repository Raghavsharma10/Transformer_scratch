def STRB(self, params):
        """
        STRB Ra, [Rb, Rc]
        STRB Ra, [Rb, #imm5]

        Store Ra into memory as a byte
        Ra, Rb, and Rc must be low registers
        """
        Ra, Rb, Rc = self.get_three_parameters(self.THREE_PARAMETER_WITH_BRACKETS, params)

        if self.is_immediate(Rc):
            self.check_arguments(low_registers=(Ra, Rb), imm5=(Rc,))

            def STRB_func():
                self.memory[self.register[Rb] + self.convert_to_integer(Rc[1:])] = (self.register[Ra] & 0xFF)
        else:
            self.check_arguments(low_registers=(Ra, Rb, Rc))

            def STRB_func():
                self.memory[self.register[Rb] + self.register[Rc]] = (self.register[Ra] & 0xFF)

        return STRB_func