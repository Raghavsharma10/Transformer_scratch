def parse(readDataInstance):
        """Returns a L{DataDirectory}-like object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: L{ReadData} object to read from.
        
        @rtype: L{DataDirectory}
        @return: The L{DataDirectory} object containing L{consts.IMAGE_NUMBEROF_DIRECTORY_ENTRIES} L{Directory} objects.
        
        @raise DirectoryEntriesLengthException: The L{ReadData} instance has an incorrect number of L{Directory} objects.
        """
        if len(readDataInstance) == consts.IMAGE_NUMBEROF_DIRECTORY_ENTRIES * 8:
            newDataDirectory = DataDirectory()
            for i in range(consts.IMAGE_NUMBEROF_DIRECTORY_ENTRIES):
                newDataDirectory[i].name.value = dirs[i]
                newDataDirectory[i].rva.value = readDataInstance.readDword()
                newDataDirectory[i].size.value = readDataInstance.readDword()
        else:
            raise excep.DirectoryEntriesLengthException("The IMAGE_NUMBEROF_DIRECTORY_ENTRIES does not match with the length of the passed argument.")
        return newDataDirectory