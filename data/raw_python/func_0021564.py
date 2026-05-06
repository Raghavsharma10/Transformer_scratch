def parse(readDataInstance):
        """
        Returns a new L{NetMetaDataTableHeader} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{NetMetaDataTableHeader} object.
        
        @rtype: L{NetMetaDataTableHeader}
        @return: A new L{NetMetaDataTableHeader} object.
        """
        th = NetMetaDataTableHeader()
        
        th.reserved_1.value = readDataInstance.readDword()
        th.majorVersion.value = readDataInstance.readByte()
        th.minorVersion.value = readDataInstance.readByte()
        th.heapOffsetSizes.value = readDataInstance.readByte()
        th.reserved_2.value = readDataInstance.readByte()
        th.maskValid.value = readDataInstance.readQword()
        th.maskSorted.value = readDataInstance.readQword()

        return th