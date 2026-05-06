def isCFGEnabled(self):
        """
        Determines if the current L{PE} instance has CFG (Control Flow Guard) flag enabled.
        @see: U{http://blogs.msdn.com/b/vcblog/archive/2014/12/08/visual-studio-2015-preview-work-in-progress-security-feature.aspx}
        @see: U{https://msdn.microsoft.com/en-us/library/dn919635%%28v=vs.140%%29.aspx}

        @rtype: bool
        @return: Returns C{True} if the current L{PE} instance has the CFG flag enabled. Otherwise, return C{False}.
        """
        return self.ntHeaders.optionalHeader.dllCharacteristics.value & consts.IMAGE_DLL_CHARACTERISTICS_GUARD_CF == consts.IMAGE_DLL_CHARACTERISTICS_GUARD_CF