def parse(readDataInstance,  nEntries):
        """
        Returns a new L{ImageImportDescriptor} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{ImageImportDescriptor} object.
        
        @type nEntries: int
        @param nEntries: The number of L{ImageImportDescriptorEntry} objects in the C{readDataInstance} object.
        
        @rtype: L{ImageImportDescriptor}
        @return: A new L{ImageImportDescriptor} object.
        
        @raise DataLengthException: If not enough data to read.
        """
        importEntries = ImageImportDescriptor()
        
        dataLength = len(readDataInstance)
        toRead = nEntries * consts.SIZEOF_IMAGE_IMPORT_ENTRY32
        if dataLength >= toRead:
            for i in range(nEntries):
                importEntry = ImageImportDescriptorEntry.parse(readDataInstance)
                importEntries.append(importEntry)
        else:
            raise excep.DataLengthException("Not enough bytes to read.")
            
        return importEntries