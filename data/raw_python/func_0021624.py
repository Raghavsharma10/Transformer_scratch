def isDll(self):
        """
        Determines if the current L{PE} instance is a Dynamic Link Library file.
        
        @rtype: bool
        @return: C{True} if the current L{PE} instance is a DLL. Otherwise, returns C{False}.
        """
        if (consts.IMAGE_FILE_DLL & self.ntHeaders.fileHeader.characteristics.value) == consts.IMAGE_FILE_DLL:
            return True
        return False