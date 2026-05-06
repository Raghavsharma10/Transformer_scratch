def parse(readDataInstance):
        """
        Returns a new L{ImageImportDescriptorEntry} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{ImageImportDescriptorEntry}.
        
        @rtype: L{ImageImportDescriptorEntry}
        @return: A new L{ImageImportDescriptorEntry} object.
        """
        iid = ImageImportDescriptorEntry()
        iid.originalFirstThunk.value = readDataInstance.readDword()
        iid.timeDateStamp.value = readDataInstance.readDword()
        iid.forwarderChain.value = readDataInstance.readDword()
        iid.name.value = readDataInstance.readDword()
        iid.firstThunk.value = readDataInstance.readDword()
        return iid