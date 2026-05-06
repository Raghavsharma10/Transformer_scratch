def parse(readDataInstance):
        """
        Returns a new L{ImageDebugDirectory} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A new L{ReadData} object with data to be parsed as a L{ImageDebugDirectory} object.
        
        @rtype: L{ImageDebugDirectory}
        @return: A new L{ImageDebugDirectory} object.
        """
        dbgDir = ImageDebugDirectory()

        dbgDir.characteristics.value = readDataInstance.readDword()
        dbgDir.timeDateStamp.value = readDataInstance.readDword()
        dbgDir.majorVersion.value = readDataInstance.readWord()
        dbgDir.minorVersion.value = readDataInstance.readWord()
        dbgDir.type.value = readDataInstance.readDword()
        dbgDir.sizeOfData.value = readDataInstance.readDword()
        dbgDir.addressOfData.value = readDataInstance.readDword()
        dbgDir.pointerToRawData.value = readDataInstance.readDword()
        
        return dbgDir