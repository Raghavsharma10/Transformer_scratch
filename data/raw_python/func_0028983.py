def RSBS(self, params):
        """
        RSBS [Ra,] Rb, #0

        Subtract Rb from zero (0 - Rb) and store the result in Ra
        Set the NZCV flags
        Ra and Rb must be low registers
        if Ra is omitted, then it is assumed to be Rb
        """
        # This instruction allows for an optional destination register
        # If it is omitted, then it is assumed to be Rb
        # As defined in http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0662b/index.html
        try:
            Ra, Rb, Rc = self.get_three_parameters(self.THREE_PARAMETER_COMMA_SEPARATED, params)
        except iarm.exceptions.ParsingError:
            Rb, Rc = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)
            Ra = Rb

        self.check_arguments(low_registers=(Ra, Rb))
        if Rc != '#0':
            raise iarm.exceptions.RuleError("Third parameter {} is not #0".format(Rc))
        # RSBS Ra, Rb, #0

        def RSBS_func():
            oper_2 = self.register[Rb]
            self.register[Ra] = 0 - self.register[Rb]
            self.set_NZCV_flags(0, oper_2, self.register[Ra], 'sub')

        return RSBS_func