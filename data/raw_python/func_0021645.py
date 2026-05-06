def parse(readDataInstance):
        """
        Returns a new L{FileHeader} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{FileHeader} object.
        
        @rtype: L{FileHeader}
        @return: A new L{ReadData} object.
        """
        fh = FileHeader()
        fh.machine.value  = readDataInstance.readWord()
        fh.numberOfSections.value  = readDataInstance.readWord()
        fh.timeDateStamp.value  = readDataInstance.readDword()
        fh.pointerToSymbolTable.value  = readDataInstance.readDword()
        fh.numberOfSymbols.value  = readDataInstance.readDword()
        fh.sizeOfOptionalHeader.value  = readDataInstance.readWord()
        fh.characteristics.value = readDataInstance.readWord()
        return fh