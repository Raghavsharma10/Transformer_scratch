def LDRB(self, params):
        """
        LDRB Ra, [Rb, Rc]
        LDRB Ra, [Rb, #imm5]

        Load a byte from memory into Ra
        Ra, Rb, and Rc must be low registers
        """
        try:
            Ra, Rb, Rc = self.get_three_parameters(self.THREE_PARAMETER_WITH_BRACKETS, params)
        except iarm.exceptions.ParsingError:
            # LDRB Rn, [Rk] translates to an offset of zero
            Ra, Rb = self.get_two_parameters(r'\s*([^\s,]*),\s*\[([^\s,]*)\](,\s*[^\s,]*)*\s*', params)
            Rc = '#0'

        if self.is_immediate(Rc):
            self.check_arguments(low_registers=(Ra, Rb), imm5=(Rc,))

            def LDRB_func():
                self.register[Ra] = self.memory[self.register[Rb] + self.convert_to_integer(Rc[1:])]
        else:
            self.check_arguments(low_registers=(Ra, Rb, Rc))

            def LDRB_func():
                self.register[Ra] = self.memory[self.register[Rb] + self.register[Rc]]

        return LDRB_func