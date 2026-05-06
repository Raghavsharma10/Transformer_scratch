def ADCS(self, params):
        """
        ADCS [Ra,] Rb, Rc

        Add Rb and Rc + the carry bit and store the result in Ra
        Ra, Rb, and Rc must be low registers
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

        self.check_arguments(low_registers=(Ra, Rc))
        self.match_first_two_parameters(Ra, Rb)

        # ADCS Ra, Ra, Rb
        def ADCS_func():
            # TODO need to rethink the set_NZCV with the C flag
            oper_1 = self.register[Ra]
            oper_2 = self.register[Rc]
            self.register[Ra] = oper_1 + oper_2
            self.register[Ra] += 1 if self.is_C_set() else 0
            self.set_NZCV_flags(oper_1, oper_2, self.register[Ra], 'add')

        return ADCS_func