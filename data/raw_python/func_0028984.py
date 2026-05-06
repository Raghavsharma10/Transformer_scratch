def SUB(self, params):
        """
        SUB [SP,] SP, #imm9_4

        Subtract an immediate from the Stack Pointer
        The first SP is optional
        """
        # This instruction allows for an optional destination register
        # If it is omitted, then it is assumed to be Rb
        # As defined in http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0662b/index.html
        try:
            Ra, Rb, Rc = self.get_three_parameters(self.THREE_PARAMETER_COMMA_SEPARATED, params)
        except iarm.exceptions.ParsingError:
            Rb, Rc = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)
            Ra = Rb

        self.check_arguments(imm9_4=(Rc,))
        if Ra != 'SP':
            raise iarm.exceptions.RuleError("First parameter {} is not equal to SP".format(Ra))
        if Rb != 'SP':
            raise iarm.exceptions.RuleError("Second parameter {} is not equal to SP".format(Rb))

        # SUB SP, SP, #imm9_4
        def SUB_func():
            self.register[Ra] = self.register[Rb] - self.convert_to_integer(Rc[1:])

        return SUB_func