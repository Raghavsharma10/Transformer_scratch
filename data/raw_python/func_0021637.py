def _parseLoadConfigDirectory(self, rva, size, magic = consts.PE32):
        """
        Parses IMAGE_LOAD_CONFIG_DIRECTORY.
        
        @type rva: int 
        @param rva: The RVA where the IMAGE_LOAD_CONFIG_DIRECTORY starts.
        
        @type size: int
        @param size: The size of the IMAGE_LOAD_CONFIG_DIRECTORY.
        
        @type magic: int
        @param magic: (Optional) The type of PE. This value could be L{consts.PE32} or L{consts.PE64}.
        
        @rtype: L{ImageLoadConfigDirectory}
        @return: A new L{ImageLoadConfigDirectory}. 
        @note: if the L{PE} instance is a PE64 file then a new L{ImageLoadConfigDirectory64} is returned.
        """
        # print "RVA: %x - SIZE: %x" % (rva, size)

        # I've found some issues when parsing the IMAGE_LOAD_CONFIG_DIRECTORY in some DLLs. 
        # There is an inconsistency with the size of the struct between MSDN docs and VS.
        # sizeof(IMAGE_LOAD_CONFIG_DIRECTORY) should be 0x40, in fact, that's the size Visual Studio put
        # in the directory table, even if the DLL was compiled with SAFESEH:ON. But If that is the case, the sizeof the
        # struct should be 0x48.
        # more information here: http://www.accuvant.com/blog/old-meets-new-microsoft-windows-safeseh-incompatibility
        data = self.getDataAtRva(rva, directories.ImageLoadConfigDirectory().sizeof())
        rd = utils.ReadData(data)

        if magic == consts.PE32:
            return directories.ImageLoadConfigDirectory.parse(rd)
        elif magic == consts.PE64:
            return directories.ImageLoadConfigDirectory64.parse(rd)
        else:
            raise excep.InvalidParameterException("Wrong magic")