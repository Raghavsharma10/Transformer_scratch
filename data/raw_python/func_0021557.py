def parse(readDataInstance):
        """
        Returns a new L{ExportTableEntry} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{ExportTableEntry} object.
        
        @rtype: L{ExportTableEntry}
        @return: A new L{ExportTableEntry} object.
        """
        exportEntry = ExportTableEntry()

        exportEntry.functionRva.value = readDataInstance.readDword()
        exportEntry.nameOrdinal.value = readDataInstance.readWord()
        exportEntry.nameRva.value = readDataInstance.readDword()
        exportEntry.name.value = readDataInstance.readString()
        return exportEntry