def _parseDelayImportDirectory(self, rva, size, magic = consts.PE32):
        """
        Parses the delay imports directory.
        
        @type rva: int 
        @param rva: The RVA where the delay imports directory starts.
        
        @type size: int
        @param size: The size of the delay imports directory.
        
        @type magic: int
        @param magic: (Optional) The type of PE. This value could be L{consts.PE32} or L{consts.PE64}.
        
        @rtype: str
        @return: The delay imports directory data.
        """
        return self.getDataAtRva(rva, size)