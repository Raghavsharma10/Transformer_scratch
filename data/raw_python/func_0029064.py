def RORS(self, params):
        """
        RORS [Ra,] Ra, Rc

        Rotate shift right Rb by Rc or imm5 and store the result in Ra
        The first two operands must be the same register
        Ra and Rc must be low registers
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

        # TODO implement this function
        # TODO figure out the last shifted bit
        # TODO figure out how to wrap bits around
        raise iarm.exceptions.NotImplementedError

        # RORS Ra, Ra, Rb
        self.check_arguments(low_registers=(Ra, Rc))
        self.match_first_two_parameters(Ra, Rb)

        def RORS_func():
            raise NotImplementedError

        return RORS_func