def parse(readDataInstance):
        """
        Returns a L{Directory}-like object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: L{ReadData} object to read from.
        
        @rtype: L{Directory}
        @return: L{Directory} object.
        """
        d = Directory()
        d.rva.value = readDataInstance.readDword()
        d.size.value = readDataInstance.readDword()
        return d