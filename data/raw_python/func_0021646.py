def parse(readDataInstance):
        """
        Returns a new L{OptionalHeader} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{OptionalHeader} object.
        
        @rtype: L{OptionalHeader}
        @return: A new L{OptionalHeader} object.
        """
        oh = OptionalHeader()

        oh.magic.value  = readDataInstance.readWord()
        oh.majorLinkerVersion.value  = readDataInstance.readByte()
        oh.minorLinkerVersion.value  = readDataInstance.readByte()
        oh.sizeOfCode.value  = readDataInstance.readDword()
        oh.sizeOfInitializedData.value  = readDataInstance.readDword()
        oh.sizeOfUninitializedData.value  = readDataInstance.readDword()
        oh.addressOfEntryPoint.value  = readDataInstance.readDword()
        oh.baseOfCode.value  = readDataInstance.readDword()
        oh.baseOfData.value  = readDataInstance.readDword()
        oh.imageBase.value  = readDataInstance.readDword()
        oh.sectionAlignment.value  = readDataInstance.readDword()
        oh.fileAlignment.value  = readDataInstance.readDword()
        oh.majorOperatingSystemVersion.value  = readDataInstance.readWord()
        oh.minorOperatingSystemVersion.value  = readDataInstance.readWord()
        oh.majorImageVersion.value  = readDataInstance.readWord()
        oh.minorImageVersion.value  = readDataInstance.readWord()
        oh.majorSubsystemVersion.value  = readDataInstance.readWord()
        oh.minorSubsystemVersion.value  = readDataInstance.readWord()
        oh.win32VersionValue.value  = readDataInstance.readDword()
        oh.sizeOfImage.value  = readDataInstance.readDword()
        oh.sizeOfHeaders.value  = readDataInstance.readDword()
        oh.checksum.value  = readDataInstance.readDword()
        oh.subsystem.value  = readDataInstance.readWord()
        oh.dllCharacteristics.value  = readDataInstance.readWord()
        oh.sizeOfStackReserve.value  = readDataInstance.readDword()
        oh.sizeOfStackCommit.value  = readDataInstance.readDword()
        oh.sizeOfHeapReserve.value  = readDataInstance.readDword()
        oh.sizeOfHeapCommit.value  = readDataInstance.readDword()
        oh.loaderFlags.value  = readDataInstance.readDword()
        oh.numberOfRvaAndSizes.value  = readDataInstance.readDword()
        
        dirs = readDataInstance.read(consts.IMAGE_NUMBEROF_DIRECTORY_ENTRIES * 8)

        oh.dataDirectory = datadirs.DataDirectory.parse(utils.ReadData(dirs))

        return oh