def parse(readDataInstance):
        """
        Returns a new L{SectionHeader} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{SectionHeader} object.
        
        @rtype: L{SectionHeader}
        @return: A new L{SectionHeader} object.
        """
        sh = SectionHeader()
        sh.name.value = readDataInstance.read(8)
        sh.misc.value  = readDataInstance.readDword()
        sh.virtualAddress.value  = readDataInstance.readDword()
        sh.sizeOfRawData.value  = readDataInstance.readDword()
        sh.pointerToRawData.value  = readDataInstance.readDword()
        sh.pointerToRelocations.value  = readDataInstance.readDword()
        sh.pointerToLineNumbers.value  = readDataInstance.readDword()
        sh.numberOfRelocations.value  = readDataInstance.readWord()
        sh.numberOfLinesNumbers.value  = readDataInstance.readWord()
        sh.characteristics.value  = readDataInstance.readDword()
        return sh