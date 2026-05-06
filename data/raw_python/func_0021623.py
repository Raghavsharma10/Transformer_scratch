def isExe(self):
        """
        Determines if the current L{PE} instance is an Executable file.
        
        @rtype: bool
        @return: C{True} if the current L{PE} instance is an Executable file. Otherwise, returns C{False}.
        """
        if not self.isDll() and not self.isDriver() and ( consts.IMAGE_FILE_EXECUTABLE_IMAGE & self.ntHeaders.fileHeader.characteristics.value) == consts.IMAGE_FILE_EXECUTABLE_IMAGE:
            return True
        return False