def getSectionByRva(self, rva):
        """
        Given a RVA in the file, tries to determine the section this RVA belongs to.
        
        @type rva: int
        @param rva: RVA value.
        
        @rtype: int
        @return: An index, starting at 1, that represents the section the given RVA belongs to.
        """
        
        index = -1
        if rva < self.sectionHeaders[0].virtualAddress.value:
            return index
        
        for i in range(len(self.sectionHeaders)):
            fa = self.ntHeaders.optionalHeader.fileAlignment.value
            prd = self.sectionHeaders[i].pointerToRawData.value
            srd = self.sectionHeaders[i].sizeOfRawData.value
            if len(str(self)) - self._adjustFileAlignment(prd,  fa) < srd:
                size = self.sectionHeaders[i].misc.value
            else:
                size = max(srd,  self.sectionHeaders[i].misc.value)
            if (self.sectionHeaders[i].virtualAddress.value <= rva) and rva < (self.sectionHeaders[i].virtualAddress.value + size):
                index = i
                break

        return index