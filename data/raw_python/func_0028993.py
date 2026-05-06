def MOVS(self, params):
        """
        MOVS Ra, Rb
        MOVS Ra, #imm8

        Move the value of Rb or imm8 into Ra
        Ra and Rb must be low registers
        """
        Ra, Rb = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)

        if self.is_immediate(Rb):
            self.check_arguments(low_registers=[Ra], imm8=[Rb])

            def MOVS_func():
                self.register[Ra] = self.convert_to_integer(Rb[1:])

                # Set N and Z status flags
                self.set_NZ_flags(self.register[Ra])

            return MOVS_func
        elif self.is_register(Rb):
            self.check_arguments(low_registers=(Ra, Rb))

            def MOVS_func():
                self.register[Ra] = self.register[Rb]

                self.set_NZ_flags(self.register[Ra])

            return MOVS_func
        else:
            raise iarm.exceptions.ParsingError("Unknown parameter: {}".format(Rb))