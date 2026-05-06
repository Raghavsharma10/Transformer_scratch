def _parseExceptionDirectory(self, rva, size, magic = consts.PE32):
        """
        Parses the C{IMAGE_EXCEPTION_DIRECTORY} directory.
        
        @type rva: int 
        @param rva: The RVA where the C{IMAGE_EXCEPTION_DIRECTORY} starts.
        
        @type size: int
        @param size: The size of the C{IMAGE_EXCEPTION_DIRECTORY} directory.
        
        @type magic: int
        @param magic: (Optional) The type of PE. This value could be L{consts.PE32} or L{consts.PE64}.
        
        @rtype: str
        @return: The C{IMAGE_EXCEPTION_DIRECTORY} data.
        """
        return self.getDataAtRva(rva, size)