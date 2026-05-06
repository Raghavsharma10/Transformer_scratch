def getSectionByOffset(self, offset):
        """
        Given an offset in the file, tries to determine the section this offset belong to.
        
        @type offset: int
        @param offset: Offset value.
        
        @rtype: int
        @return: An index, starting at 0, that represents the section the given offset belongs to.
        """
        index = -1
        for i in range(len(self.sectionHeaders)):
            if (offset < self.sectionHeaders[i].pointerToRawData.value + self.sectionHeaders[i].sizeOfRawData.value):
                index = i
                break
        return index