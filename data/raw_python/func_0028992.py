def MOV(self, params):
        """
        MOV Rx, Ry
        MOV PC, Ry

        Move the value of Ry into Rx or PC
        """
        Rx, Ry = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)

        self.check_arguments(any_registers=(Rx, Ry))

        def MOV_func():
            self.register[Rx] = self.register[Ry]

        return MOV_func