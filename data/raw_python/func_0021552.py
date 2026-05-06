def parse(readDataInstance):
        """
        Returns a new L{ImageBaseRelocationEntry} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to parse as a L{ImageBaseRelocationEntry} object.
        
        @rtype: L{ImageBaseRelocationEntry}
        @return: A new L{ImageBaseRelocationEntry} object.
        """
        reloc = ImageBaseRelocationEntry()
        reloc.virtualAddress.value = readDataInstance.readDword()
        reloc.sizeOfBlock.value = readDataInstance.readDword()
        toRead = (reloc.sizeOfBlock.value - 8) / len(datatypes.WORD(0))
        reloc.items = datatypes.Array.parse(readDataInstance,  datatypes.TYPE_WORD,  toRead)
        return reloc