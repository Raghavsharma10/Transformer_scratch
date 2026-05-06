def getDataAtOffset(self, offset, size):
        """
        Gets binary data at a given offset.
        
        @type offset: int
        @param offset: The offset to get the data from.
        
        @type size: int
        @param size: The size of the data to be obtained.
        
        @rtype: str
        @return: The data obtained at the given offset.
        """
        data = str(self)
        return data[offset:offset+size]