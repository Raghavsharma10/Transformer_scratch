def STR(self, params):
        """
        STR Ra, [Rb, Rc]
        STR Ra, [Rb, #imm7_4]
        STR Ra, [SP, #imm10_4]

        Store Ra into memory as a word
        Ra, Rb, and Rc must be low registers
        """
        Ra, Rb, Rc = self.get_three_parameters(self.THREE_PARAMETER_WITH_BRACKETS, params)

        if self.is_immediate(Rc):
            if Rb == 'SP' or Rb == 'FP':
                self.check_arguments(low_registers=(Ra,), imm10_4=(Rc,))
            else:
                self.check_arguments(low_registers=(Ra, Rb), imm7_4=(Rc,))

            def STR_func():
                for i in range(4):
                    self.memory[self.register[Rb] + self.convert_to_integer(Rc[1:]) + i] = ((self.register[Ra] >> (8 * i)) & 0xFF)
        else:
            self.check_arguments(low_registers=(Ra, Rb, Rc))

            def STR_func():
                for i in range(4):
                    self.memory[self.register[Rb] + self.register[Rc] + i] = ((self.register[Ra] >> (8 * i)) & 0xFF)

        return STR_func