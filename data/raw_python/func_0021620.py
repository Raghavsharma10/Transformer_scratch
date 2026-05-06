def getDataAtRva(self, rva, size):
        """
        Gets binary data at a given RVA.
        
        @type rva: int
        @param rva: The RVA to get the data from.
        
        @type size: int
        @param size: The size of the data to be obtained. 
        
        @rtype: str
        @return: The data obtained at the given RVA.
        """
        return self.getDataAtOffset(self.getOffsetFromRva(rva),  size)