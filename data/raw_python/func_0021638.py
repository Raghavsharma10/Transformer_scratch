def _parseTlsDirectory(self, rva, size, magic = consts.PE32):
        """
        Parses the TLS directory.
        
        @type rva: int 
        @param rva: The RVA where the TLS directory starts.
        
        @type size: int
        @param size: The size of the TLS directory.
        
        @type magic: int
        @param magic: (Optional) The type of PE. This value could be L{consts.PE32} or L{consts.PE64}.
        
        @rtype: L{TLSDirectory}
        @return: A new L{TLSDirectory}. 
        @note: if the L{PE} instance is a PE64 file then a new L{TLSDirectory64} is returned.
        """
        data = self.getDataAtRva(rva, size)
        rd = utils.ReadData(data)
        
        if magic == consts.PE32:
            return directories.TLSDirectory.parse(rd)
        elif magic == consts.PE64:
            return directories.TLSDirectory64.parse(rd)
        else:
            raise excep.InvalidParameterException("Wrong magic")