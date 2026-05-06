def CMP(self, params):
        """
        CMP Rm, Rn
        CMP Rm, #imm8

        Subtract Rn or imm8 from Rm, set the NZCV flags, and discard the result
        Rm and Rn can be R0-R14
        """
        Rm, Rn = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)

        if self.is_register(Rn):
            # CMP Rm, Rn
            self.check_arguments(R0_thru_R14=(Rm, Rn))

            def CMP_func():
                self.set_NZCV_flags(self.register[Rm], self.register[Rn],
                                    self.register[Rm] - self.register[Rn], 'sub')
        else:
            # CMP Rm, #imm8
            self.check_arguments(R0_thru_R14=(Rm,), imm8=(Rn,))

            def CMP_func():
                tmp = self.convert_to_integer(Rn[1:])
                self.set_NZCV_flags(self.register[Rm], tmp,
                                    self.register[Rm] - tmp, 'sub')

        return CMP_func