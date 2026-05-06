def isPe32(self):
        """
        Determines if the current L{PE} instance is a PE32 file.
        
        @rtype: bool
        @return: C{True} if the current L{PE} instance is a PE32 file. Otherwise, returns C{False}.
        """
        if self.ntHeaders.optionalHeader.magic.value == consts.PE32:
            return True
        return False