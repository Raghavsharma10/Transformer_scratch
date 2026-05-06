def parse(readDataInstance):
        """
        Returns a new L{NtHeaders} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{NtHeaders} object.
        
        @rtype: L{NtHeaders}
        @return: A new L{NtHeaders} object.
        """
        nt = NtHeaders()
        nt.signature.value = readDataInstance.readDword()
        nt.fileHeader = FileHeader.parse(readDataInstance)
        nt.optionalHeader = OptionalHeader.parse(readDataInstance)
        return nt