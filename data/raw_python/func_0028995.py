def MSR(self, params):
        """
        MSR Rspecial, Rj

        Copy the value of Rj to Rspecial
        Rspecial can be APSR, IPSR, or EPSR
        """
        Rspecial, Rj = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)

        self.check_arguments(LR_or_general_purpose_registers=(Rj,), special_registers=(Rspecial,))

        def MSR_func():
            # TODO add combination registers IEPSR, IAPSR, and EAPSR
            # http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0553a/CHDBIBGJ.html
            # TODO update N Z C V flags
            if Rspecial in ('PSR', 'APSR'):
                # PSR ignores writes to IPSR and EPSR
                self.register['APSR'] = self.register[Rj]
            else:
                # Do nothing
                pass

        return MSR_func