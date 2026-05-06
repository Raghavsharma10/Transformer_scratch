def parse(readDataInstance):
        """
        Returns a new L{NetMetaDataStreamEntry} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{NetMetaDataStreamEntry}.
        
        @rtype: L{NetMetaDataStreamEntry}
        @return: A new L{NetMetaDataStreamEntry} object.
        """
        n = NetMetaDataStreamEntry()
        n.offset.value = readDataInstance.readDword()
        n.size.value = readDataInstance.readDword()
        n.name.value = readDataInstance.readAlignedString()
        return n