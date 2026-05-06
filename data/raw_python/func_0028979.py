def ADD(self, params):
        """
        ADD [Rx,] Ry, [Rz, PC]
        ADD [Rx,] [SP, PC], #imm10_4
        ADD [SP,] SP, #imm9_4

        Add Ry and Rz and store the result in Rx
        Rx, Ry, and Rz can be any register
        If Rx is omitted, then it is assumed to be Ry
        """
        # This instruction allows for an optional destination register
        # If it is omitted, then it is assumed to be Rb
        # As defined in http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0662b/index.html
        # TODO can we have ADD SP, #imm9_4?
        try:
            Rx, Ry, Rz = self.get_three_parameters(self.THREE_PARAMETER_COMMA_SEPARATED, params)
        except iarm.exceptions.ParsingError:
            Ry, Rz = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)
            Rx = Ry

        if self.is_register(Rz):
            # ADD Rx, Ry, Rz
            self.check_arguments(any_registers=(Rx, Ry, Rz))
            if Rx != Ry:
                raise iarm.exceptions.RuleError("Second parameter {} does not equal first parameter {}". format(Ry, Rx))

            def ADD_func():
                self.register[Rx] = self.register[Ry] + self.register[Rz]
        else:
            if Rx == 'SP':
                # ADD SP, SP, #imm9_4
                self.check_arguments(imm9_4=(Rz,))
                if Rx != Ry:
                    raise iarm.exceptions.RuleError("Second parameter {} is not SP".format(Ry))
            else:
                # ADD Rx, [SP, PC], #imm10_4
                self.check_arguments(any_registers=(Rx,), imm10_4=(Rz,))
                if Ry not in ('SP', 'PC'):
                    raise iarm.exceptions.RuleError("Second parameter {} is not SP or PC".format(Ry))

            def ADD_func():
                self.register[Rx] = self.register[Ry] + self.convert_to_integer(Rz[1:])

        return ADD_func