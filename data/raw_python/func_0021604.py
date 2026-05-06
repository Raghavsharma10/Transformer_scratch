def getOffsetFromRva(self, rva):
        """
        Converts an offset to an RVA.
        
        @type rva: int
        @param rva: The RVA to be converted.
        
        @rtype: int
        @return: An integer value representing an offset in the PE file.
        """
        offset = -1
        s = self.getSectionByRva(rva)
        
        if s != offset:
            offset = (rva - self.sectionHeaders[s].virtualAddress.value) + self.sectionHeaders[s].pointerToRawData.value
        else:
            offset = rva
        
        return offset