def _parseRelocsDirectory(self, rva, size, magic = consts.PE32):
        """
        Parses the relocation directory.
        
        @type rva: int 
        @param rva: The RVA where the relocation directory starts.
        
        @type size: int
        @param size: The size of the relocation directory.
        
        @type magic: int
        @param magic: (Optional) The type of PE. This value could be L{consts.PE32} or L{consts.PE64}.
        
        @rtype: L{ImageBaseRelocation}
        @return: A new L{ImageBaseRelocation} object.
        """
        data = self.getDataAtRva(rva,  size)
        #print "Length Relocation data: %x" % len(data)
        rd = utils.ReadData(data)
        
        relocsArray = directories.ImageBaseRelocation()
        while rd.offset < size:
            relocEntry = directories.ImageBaseRelocationEntry.parse(rd)
            relocsArray.append(relocEntry)
        return relocsArray