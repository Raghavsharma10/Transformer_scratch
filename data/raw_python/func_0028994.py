def MRS(self, params):
        """
        MRS Rj, Rspecial

        Copy the value of Rspecial to Rj
        Rspecial can be APSR, IPSR, or EPSR
        """
        Rj, Rspecial = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)

        self.check_arguments(LR_or_general_purpose_registers=(Rj,), special_registers=(Rspecial,))

        def MRS_func():
            # TODO add combination registers IEPSR, IAPSR, and EAPSR
            # TODO needs to use APSR, IPSR, EPSR, IEPSR, IAPSR, EAPSR, PSR, MSP, PSP, PRIMASK, or CONTROL.
            # http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0553a/CHDBIBGJ.html
            if Rspecial == 'PSR':
                self.register[Rj] = self.register['APSR'] | self.register['IPSR'] | self.register['EPSR']
            else:
                self.register[Rj] = self.register[Rspecial]

        return MRS_func