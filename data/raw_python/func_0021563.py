def parse(readDataInstance,  nStreams):
        """
        Returns a new L{NetMetaDataStreams} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{NetMetaDataStreams} object.
        
        @type nStreams: int
        @param nStreams: The number of L{NetMetaDataStreamEntry} objects in the C{readDataInstance} object.
        
        @rtype: L{NetMetaDataStreams}
        @return: A new L{NetMetaDataStreams} object.
        """
        streams = NetMetaDataStreams()
        
        for i in range(nStreams):
            streamEntry = NetMetaDataStreamEntry()
            
            streamEntry.offset.value = readDataInstance.readDword()
            streamEntry.size.value = readDataInstance.readDword()
            streamEntry.name.value = readDataInstance.readAlignedString()
            
            #streams.append(streamEntry)
            streams.update({ i: streamEntry, streamEntry.name.value: streamEntry })

        return streams