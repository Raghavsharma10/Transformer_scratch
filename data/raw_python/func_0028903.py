def LDRH(self, params):
        """
        LDRH Ra, [Rb, Rc]
        LDRH Ra, [Rb, #imm6_2]

        Load a half word from memory into Ra
        Ra, Rb, and Rc must be low registers
        """
        try:
            Ra, Rb, Rc = self.get_three_parameters(self.THREE_PARAMETER_WITH_BRACKETS, params)
        except iarm.exceptions.ParsingError:
            # LDRB Rn, [Rk] translates to an offset of zero
            Ra, Rb = self.get_two_parameters(r'\s*([^\s,]*),\s*\[([^\s,]*)\](,\s*[^\s,]*)*\s*', params)
            Rc = '#0'

        if self.is_immediate(Rc):
            self.check_arguments(low_registers=(Ra, Rb), imm6_2=(Rc,))

            def LDRH_func():
                # TODO does memory read up?
                if (self.register[Rb]) % 2 != 0:
                    raise iarm.exceptions.HardFault(
                        "Memory access not half word aligned; Register: {}  Immediate: {}".format(self.register[Rb],
                                                                                                  self.convert_to_integer(
                                                                                                      Rc[1:])))
                self.register[Ra] = 0
                for i in range(2):
                    self.register[Ra] |= (self.memory[self.register[Rb] + self.convert_to_integer(Rc[1:]) + i] << (8 * i))
        else:
            self.check_arguments(low_registers=(Ra, Rb, Rc))

            def LDRH_func():
                # TODO does memory read up?
                if (self.register[Rb] + self.register[Rc]) % 2 != 0:
                    raise iarm.exceptions.HardFault(
                        "Memory access not half word aligned; Register: {}  Immediate: {}".format(self.register[Rb],
                                                                                                  self.register[Rc]))
                self.register[Ra] = 0
                for i in range(2):
                    self.register[Ra] |= (self.memory[self.register[Rb] + self.register[Rc] + i] << (8 * i))

        return LDRH_func