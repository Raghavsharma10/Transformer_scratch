def isASLREnabled(self):
        """
        Determines if the current L{PE} instance has the DYNAMICBASE (Use address space layout randomization) flag enabled.
        @see: U{http://msdn.microsoft.com/en-us/library/bb384887.aspx}

        @rtype: bool
        @return: Returns C{True} if the current L{PE} instance has the DYNAMICBASE flag enabled. Otherwise, returns C{False}.
        """
        return self.ntHeaders.optionalHeader.dllCharacteristics.value & consts.IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE == consts.IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE