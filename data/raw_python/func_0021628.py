def isPeBounded(self):
        """
        Determines if the current L{PE} instance is bounded, i.e. has a C{BOUND_IMPORT_DIRECTORY}.
        
        @rtype: bool
        @return: Returns C{True} if the current L{PE} instance is bounded. Otherwise, returns C{False}.
        """
        boundImportsDir = self.ntHeaders.optionalHeader.dataDirectory[consts.BOUND_IMPORT_DIRECTORY]
        if boundImportsDir.rva.value and boundImportsDir.size.value:
            return True
        return False