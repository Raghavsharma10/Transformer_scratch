def _parseBoundImportDirectory(self, rva, size, magic = consts.PE32):
        """
        Parses the bound import directory.
        
        @type rva: int 
        @param rva: The RVA where the bound import directory starts.
        
        @type size: int
        @param size: The size of the bound import directory.
        
        @type magic: int
        @param magic: (Optional) The type of PE. This value could be L{consts.PE32} or L{consts.PE64}.
        
        @rtype: L{ImageBoundImportDescriptor}
        @return: A new L{ImageBoundImportDescriptor} object.
        """
        data = self.getDataAtRva(rva, size)
        rd = utils.ReadData(data)
        boundImportDirectory = directories.ImageBoundImportDescriptor.parse(rd)
        
        # parse the name of every bounded import.
        for i in range(len(boundImportDirectory) - 1):
            if hasattr(boundImportDirectory[i],  "forwarderRefsList"):
                if boundImportDirectory[i].forwarderRefsList:
                    for forwarderRefEntry in boundImportDirectory[i].forwarderRefsList:
                        offset = forwarderRefEntry.offsetModuleName.value
                        forwarderRefEntry.moduleName = self.readStringAtRva(offset + rva)
                        
            offset = boundImportDirectory[i].offsetModuleName.value
            boundImportDirectory[i].moduleName = self.readStringAtRva(offset + rva)
        return boundImportDirectory