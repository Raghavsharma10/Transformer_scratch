def isPe64(self):
        """
        Determines if the current L{PE} instance is a PE64 file.
        
        @rtype: bool
        @return: C{True} if the current L{PE} instance is a PE64 file. Otherwise, returns C{False}.
        """
        if self.ntHeaders.optionalHeader.magic.value == consts.PE64:
            return True
        return False