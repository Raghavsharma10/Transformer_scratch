def parse(readDataInstance):
        """
        Returns a new L{NETDirectory} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{NETDirectory} object.
        
        @rtype: L{NETDirectory}
        @return: A new L{NETDirectory} object.
        """
        nd = NETDirectory()
        
        nd.directory = NetDirectory.parse(readDataInstance)
        nd.netMetaDataHeader = NetMetaDataHeader.parse(readDataInstance)
        nd.netMetaDataStreams = NetMetaDataStreams.parse(readDataInstance)
        return nd