def parse(readDataInstance,  numberOfEntries):
        """
        Returns a L{ImageBoundForwarderRef} array where every element is a L{ImageBoundForwarderRefEntry} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with the corresponding data to generate a new L{ImageBoundForwarderRef} object.
        
        @type numberOfEntries: int
        @param numberOfEntries: The number of C{IMAGE_BOUND_FORWARDER_REF} entries in the array.
        
        @rtype: L{ImageBoundForwarderRef}
        @return: A new L{ImageBoundForwarderRef} object.
        
        @raise DataLengthException: If the L{ReadData} instance has less data than C{NumberOfEntries} * sizeof L{ImageBoundForwarderRefEntry}.
        """
        imageBoundForwarderRefsList = ImageBoundForwarderRef()
        dLength = len(readDataInstance)
        entryLength = ImageBoundForwarderRefEntry().sizeof()
        toRead = numberOfEntries * entryLength
        
        if dLength >= toRead:
            for i in range(numberOfEntries):
                entryData = readDataInstance.read(entryLength)
                rd = utils.ReadData(entryData)
                imageBoundForwarderRefsList.append(ImageBoundForwarderRefEntry.parse(rd))
        else:
            raise excep.DataLengthException("Not enough bytes to read.")
        
        return imageBoundForwarderRefsList