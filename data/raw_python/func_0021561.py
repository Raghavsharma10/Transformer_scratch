def parse(readDataInstance):
        """
        Returns a new L{NetMetaDataHeader} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{NetMetaDataHeader} object.
        
        @rtype: L{NetMetaDataHeader}
        @return: A new L{NetMetaDataHeader} object.
        """
        nmh = NetMetaDataHeader()
        
        nmh.signature.value = readDataInstance.readDword()
        nmh.majorVersion.value = readDataInstance.readWord()
        nmh.minorVersion.value = readDataInstance.readWord()
        nmh.reserved.value = readDataInstance.readDword()
        nmh.versionLength.value = readDataInstance.readDword()
        nmh.versionString.value = readDataInstance.readAlignedString()
        nmh.flags.value = readDataInstance.readWord()
        nmh.numberOfStreams.value = readDataInstance.readWord()
        return nmh