def parse(readDataInstance,  nDebugEntries):
        """
        Returns a new L{ImageDebugDirectories} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{ImageDebugDirectories} object.
        
        @type nDebugEntries: int
        @param nDebugEntries: Number of L{ImageDebugDirectory} objects in the C{readDataInstance} object.
        
        @rtype: L{ImageDebugDirectories}
        @return: A new L{ImageDebugDirectories} object.
        
        @raise DataLengthException: If not enough data to read in the C{readDataInstance} object.
        """
        dbgEntries = ImageDebugDirectories()
        
        dataLength = len(readDataInstance)
        toRead = nDebugEntries * consts.SIZEOF_IMAGE_DEBUG_ENTRY32
        if dataLength >= toRead:
            for i in range(nDebugEntries):
                dbgEntry = ImageDebugDirectory.parse(readDataInstance)
                dbgEntries.append(dbgEntry)
        else:
            raise excep.DataLengthException("Not enough bytes to read.")
        
        return dbgEntries