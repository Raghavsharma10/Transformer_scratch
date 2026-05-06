def _parseDebugDirectory(self, rva, size, magic = consts.PE32):
        """
        Parses the C{IMAGE_DEBUG_DIRECTORY} directory.
        @see: U{http://msdn.microsoft.com/es-es/library/windows/desktop/ms680307(v=vs.85).aspx}
        
        @type rva: int 
        @param rva: The RVA where the C{IMAGE_DEBUG_DIRECTORY} directory starts.
        
        @type size: int
        @param size: The size of the C{IMAGE_DEBUG_DIRECTORY} directory.
        
        @type magic: int
        @param magic: (Optional) The type of PE. This value could be L{consts.PE32} or L{consts.PE64}.
        
        @rtype: L{ImageDebugDirectory}
        @return: A new L{ImageDebugDirectory} object.
        """        
        debugDirData = self.getDataAtRva(rva, size)
        numberOfEntries = size / consts.SIZEOF_IMAGE_DEBUG_ENTRY32
        rd = utils.ReadData(debugDirData)
        return directories.ImageDebugDirectories.parse(rd,  numberOfEntries)