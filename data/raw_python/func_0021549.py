def parse(readDataInstance):
        """
        Returns a new L{TLSDirectory} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object containing data to create a new L{TLSDirectory} object.
        
        @rtype: L{TLSDirectory}
        @return: A new {TLSDirectory} object.
        """
        tlsDir = TLSDirectory()
        
        tlsDir.startAddressOfRawData.value = readDataInstance.readDword()
        tlsDir.endAddressOfRawData.value = readDataInstance.readDword()
        tlsDir.addressOfIndex.value = readDataInstance.readDword()
        tlsDir.addressOfCallbacks.value = readDataInstance.readDword()
        tlsDir.sizeOfZeroFill.value = readDataInstance.readDword()
        tlsDir.characteristics.value = readDataInstance.readDword()
        return tlsDir