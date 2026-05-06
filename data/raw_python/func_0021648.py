def parse(readDataInstance,  numberOfSectionHeaders):
        """
        Returns a new L{SectionHeaders} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{SectionHeaders} object.
        
        @type numberOfSectionHeaders: int
        @param numberOfSectionHeaders: The number of L{SectionHeader} objects in the L{SectionHeaders} instance.
        """
        sHdrs = SectionHeaders(numberOfSectionHeaders = 0)
        
        for i in range(numberOfSectionHeaders):
            sh = SectionHeader()
            
            sh.name.value = readDataInstance.read(8)
            sh.misc.value = readDataInstance.readDword()
            sh.virtualAddress.value = readDataInstance.readDword()
            sh.sizeOfRawData.value = readDataInstance.readDword()
            sh.pointerToRawData.value = readDataInstance.readDword()
            sh.pointerToRelocations.value = readDataInstance.readDword()
            sh.pointerToLineNumbers.value = readDataInstance.readDword()
            sh.numberOfRelocations.value = readDataInstance.readWord()
            sh.numberOfLinesNumbers.value = readDataInstance.readWord()
            sh.characteristics.value = readDataInstance.readDword()
        
            sHdrs.append(sh)
        
        return sHdrs