def LSRS(self, params):
        """
        LSRS [Ra,] Ra, Rc
        LSRS [Ra,] Rb, #imm5_counting

        Logical shift right Rb by Rc or imm5 and store the result in Ra
        imm5 counting is [1, 32]
        In the register shift, the first two operands must be the same register
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
            # LSRS Ra, Ra, Rb
            self.check_arguments(low_registers=(Ra, Rc))
            self.match_first_two_parameters(Ra, Rb)

            def LSRS_func():
                # Set the C flag, or the last shifted out bit
                if (self.register[Rc] > 0) and (self.register[Rb] & (1 << (self.register[Rc] - 1))):
                    self.set_APSR_flag_to_value('C', 1)
                else:
                    self.set_APSR_flag_to_value('C', 0)

                self.register[Ra] = self.register[Ra] >> self.register[Rc]
                self.set_NZ_flags(self.register[Ra])
        else:
            # LSRS Ra, Rb, #imm5_counting
            self.check_arguments(low_registers=(Ra, Rb), imm5_counting=(Rc,))
            shift_amount = self.check_immediate(Rc)

            def LSRS_func():
                # Set the C flag, or the last shifted out bit
                if self.register[Rb] & (1 << (shift_amount - 1)):
                    self.set_APSR_flag_to_value('C', 1)
                else:
                    self.set_APSR_flag_to_value('C', 0)

                self.register[Ra] = self.register[Rb] >> shift_amount
                self.set_NZ_flags(self.register[Ra])

        return LSRS_func