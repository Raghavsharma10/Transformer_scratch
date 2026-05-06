def SUBS(self, params):
        """
        SUBS [Ra,] Rb, Rc
        SUBS [Ra,] Rb, #imm3
        SUBS [Ra,] Ra, #imm8

        Subtract Rc or an immediate from Rb and store the result in Ra
        Ra, Rb, and Rc must be low registers
        If Ra is omitted, then it is assumed to be Rb
        """
        # This instruction allows for an optional destination register
        # If it is omitted, then it is assumed to be Rb
        # As defined in http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0662b/index.html
        try:
            Ra, Rb, Rc = self.get_three_parameters(self.THREE_PARAMETER_COMMA_SEPARATED, params)
        except iarm.exceptions.ParsingError:
            Rb, Rc = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)
            Ra = Rb

        if self.is_register(Rc):
            # SUBS Ra, Rb, Rc
            self.check_arguments(low_registers=(Ra, Rb, Rc))

            def SUBS_func():
                oper_1 = self.register[Rb]
                oper_2 = self.register[Rc]
                self.register[Ra] = self.register[Rb] - self.register[Rc]
                self.set_NZCV_flags(oper_1, oper_2, self.register[Ra], 'sub')
        else:
            if Ra == Rb:
                # SUBS Ra, Ra, #imm8
                self.check_arguments(low_registers=(Ra,), imm8=(Rc,))

                def SUBS_func():
                    oper_1 = self.register[Ra]
                    self.register[Ra] = self.register[Ra] - self.convert_to_integer(Rc[1:])
                    self.set_NZCV_flags(oper_1, self.convert_to_integer(Rc[1:]), self.register[Ra], 'sub')
            else:
                # SUBS Ra, Rb, #imm3
                self.check_arguments(low_registers=(Ra, Rb), imm3=(Rc,))

                def SUBS_func():
                    oper_1 = self.register[Rb]
                    self.register[Ra] = self.register[Rb] - self.convert_to_integer(Rc[1:])
                    self.set_NZCV_flags(oper_1, self.convert_to_integer(Rc[1:]), self.register[Ra], 'sub')

        return SUBS_func