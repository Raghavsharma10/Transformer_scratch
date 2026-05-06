def isNXEnabled(self):
        """
        Determines if the current L{PE} instance has the NXCOMPAT (Compatible with Data Execution Prevention) flag enabled.
        @see: U{http://msdn.microsoft.com/en-us/library/ms235442.aspx}

        @rtype: bool
        @return: Returns C{True} if the current L{PE} instance has the NXCOMPAT flag enabled. Otherwise, returns C{False}.
        """
        return self.ntHeaders.optionalHeader.dllCharacteristics.value & consts.IMAGE_DLL_CHARACTERISTICS_NX_COMPAT == consts.IMAGE_DLL_CHARACTERISTICS_NX_COMPAT