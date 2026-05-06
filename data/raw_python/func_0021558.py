def parse(readDataInstance):
        """
        Returns a new L{ImageExportTable} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{ImageExportTable} object.
        
        @rtype: L{ImageExportTable}
        @return: A new L{ImageExportTable} object.
        """
        et = ImageExportTable()
        
        et.characteristics.value = readDataInstance.readDword()
        et.timeDateStamp.value = readDataInstance.readDword()
        et.majorVersion.value = readDataInstance.readWord()
        et.minorVersion.value = readDataInstance.readWord()
        et.name.value = readDataInstance.readDword()
        et.base.value = readDataInstance.readDword()
        et.numberOfFunctions.value = readDataInstance.readDword()
        et.numberOfNames.value = readDataInstance.readDword()
        et.addressOfFunctions.value = readDataInstance.readDword()
        et.addressOfNames.value = readDataInstance.readDword()
        et.addressOfNameOrdinals.value = readDataInstance.readDword()
        return et