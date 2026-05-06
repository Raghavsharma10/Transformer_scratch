def _getPaddingDataToSectionOffset(self):
        """
        Returns the data between the last section header and the begenning of data from the first section.
        
        @rtype: str
        @return: Data between last section header and the begenning of the first section.
        """
        start = self._getPaddingToSectionOffset()
        end = self.sectionHeaders[0].pointerToRawData.value - start
        return self._data[start:start+end]