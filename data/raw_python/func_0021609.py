def _getPaddingToSectionOffset(self):
        """
        Returns the offset to last section header present in the PE file.
        
        @rtype: int
        @return: The offset where the end of the last section header resides in the PE file.
        """
        return len(str(self.dosHeader) + str(self.dosStub) + str(self.ntHeaders) + str(self.sectionHeaders))