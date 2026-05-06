def getRvaFromOffset(self, offset):
        """
        Converts a RVA to an offset.
        
        @type offset: int
        @param offset: The offset value to be converted to RVA.
        
        @rtype: int
        @return: The RVA obtained from the given offset.
        """
        rva = -1
        s = self.getSectionByOffset(offset)
        
        if s:
            rva = (offset - self.sectionHeaders[s].pointerToRawData.value) + self.sectionHeaders[s].virtualAddress.value
            
        return rva