def ADR(self, params):
        """
        ADR Ra, [PC, #imm10_4]
        ADR Ra, label

        Load the address of label or the PC offset into Ra
        Ra must be a low register
        """
        # TODO may need to rethink how I do PC, may need to be byte alligned
        # TODO This is wrong as each address is a word, not a byte. The filled value with its location (Do we want that, or the value at that location [Decompiled instruction])
        try:
            Ra, Rb, Rc = self.get_three_parameters(self.THREE_PARAMETER_WITH_BRACKETS, params)
        except iarm.exceptions.ParsingError:
            Ra, label = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)

            # TODO the address must be within 1020 bytes of current PC
            self.check_arguments(low_registers=(Ra,), label_exists=(label,))

            def ADR_func():
                self.register[Ra] = self.labels[label]  # TODO is this correct?

            return ADR_func

        self.check_arguments(low_registers=(Ra,), imm10_4=(Rc,))
        if Rb != 'PC':
            raise iarm.exceptions.IarmError("Second position argument is not PC: {}".format(Rb))

        def ADR_func():
            self.register[Ra] = self.register[Rb] + self.convert_to_integer(Rc[1:])

        return ADR_func