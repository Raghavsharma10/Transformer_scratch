def parse(readDataInstance):
        """
        Returns a new L{ImageBoundForwarderRefEntry} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with the corresponding data to generate a new L{ImageBoundForwarderRefEntry} object.
        
        @rtype: L{ImageBoundForwarderRefEntry}
        @return: A new L{ImageBoundForwarderRefEntry} object.
        """
        boundForwarderEntry = ImageBoundForwarderRefEntry()
        boundForwarderEntry.timeDateStamp.value = readDataInstance.readDword()
        boundForwarderEntry.offsetModuleName.value = readDataInstance.readWord()
        boundForwarderEntry.reserved.value = readDataInstance.readWord()
        return boundForwarderEntry