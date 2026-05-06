def parse(readDataInstance):
        """
        Returns a new L{TLSDirectory64} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object containing data to create a new L{TLSDirectory64} object.
        
        @rtype: L{TLSDirectory64}
        @return: A new L{TLSDirectory64} object.
        """
        tlsDir = TLSDirectory64()
        
        tlsDir.startAddressOfRawData.value = readDataInstance.readQword()
        tlsDir.endAddressOfRawData.value = readDataInstance.readQword()
        tlsDir.addressOfIndex.value = readDataInstance.readQword()
        tlsDir.addressOfCallbacks.value = readDataInstance.readQword()
        tlsDir.sizeOfZeroFill.value = readDataInstance.readDword()
        tlsDir.characteristics.value = readDataInstance.readDword()
        return tlsDir