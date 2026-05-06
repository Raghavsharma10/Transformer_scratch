def isSAFESEHEnabled(self):
        """
        Determines if the current L{PE} instance has the SAFESEH (Image has Safe Exception Handlers) flag enabled.
        @see: U{http://msdn.microsoft.com/en-us/library/9a89h429.aspx}

        @rtype: bool
        @return: Returns C{True} if the current L{PE} instance has the SAFESEH flag enabled. Returns C{False} if SAFESEH is off or -1 if SAFESEH is set to NO.
        """
        NOSEH = -1
        SAFESEH_OFF = 0
        SAFESEH_ON = 1

        if self.ntHeaders.optionalHeader.dllCharacteristics.value & consts.IMAGE_DLL_CHARACTERISTICS_NO_SEH:
            return NOSEH

        loadConfigDir = self.ntHeaders.optionalHeader.dataDirectory[consts.CONFIGURATION_DIRECTORY]
        if loadConfigDir.info:
            if loadConfigDir.info.SEHandlerTable.value:
                return SAFESEH_ON
        return SAFESEH_OFF