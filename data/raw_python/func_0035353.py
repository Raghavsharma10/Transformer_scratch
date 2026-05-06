def setdbo(self, bond1, bond2, dboval):
        """Set the double bond orientation for bond1 and bond2
        based on this bond"""
        # this bond must be a double bond
        if self.bondtype != 2:
            raise FrownsError("To set double bond order, center bond must be double!")
        assert dboval in [DX_CHI_CIS, DX_CHI_TRANS, DX_CHI_NO_DBO], "bad dboval value"
        
        self.dbo.append(bond1, bond2, dboval)