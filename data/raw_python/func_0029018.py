def ORRS(self, params):
        """
        ORRS [Ra,] Ra, Rb

        OR Ra and Rb together and store the result in Ra
        The equivalent of `Ra = Ra | Rc`
        Updates NZ flags
        Ra and Rb must be low registers
        The first register is optional
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

        # ORRS Ra, Ra, Rb
        def ORRS_func():
            self.register[Ra] = self.register[Ra] | self.register[Rc]
            self.set_NZ_flags(self.register[Ra])

        return ORRS_func