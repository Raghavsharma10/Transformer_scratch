def parse(readDataInstance):
        """
        Returns a new L{ImageLoadConfigDirectory64} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object containing data to create a new L{ImageLoadConfigDirectory64} object.
        
        @rtype: L{ImageLoadConfigDirectory64}
        @return: A new L{ImageLoadConfigDirectory64} object.
        """
        configDir = ImageLoadConfigDirectory64()

        configDir.size.value = readDataInstance.readDword()
        configDir.timeDateStamp.value = readDataInstance.readDword()
        configDir.majorVersion.value = readDataInstance.readWord()
        configDir.minorVersion.value = readDataInstance.readWord()
        configDir.globalFlagsClear.value = readDataInstance.readDword()
        configDir.globalFlagsSet.value = readDataInstance.readDword()
        configDir.criticalSectionDefaultTimeout.value = readDataInstance.readDword()
        configDir.deCommitFreeBlockThreshold.value = readDataInstance.readQword()
        configDir.deCommitTotalFreeThreshold.value = readDataInstance.readQword()
        configDir.lockPrefixTable.value = readDataInstance.readQword()
        configDir.maximumAllocationSize.value = readDataInstance.readQword()
        configDir.virtualMemoryThreshold.value = readDataInstance.readQword()
        configDir.processAffinityMask.value = readDataInstance.readQword()
        configDir.processHeapFlags.value = readDataInstance.readDword()
        configDir.cdsVersion.value = readDataInstance.readWord()
        configDir.reserved1.value = readDataInstance.readWord()
        configDir.editList.value = readDataInstance.readQword()
        configDir.securityCookie.value = readDataInstance.readQword()
        configDir.SEHandlerTable.value = readDataInstance.readQword()
        configDir.SEHandlerCount.value = readDataInstance.readQword()

        # Fields for Control Flow Guard
        configDir.GuardCFCheckFunctionPointer.value = readDataInstance.readQword() # VA
        configDir.Reserved2.value = readDataInstance.readQword()
        configDir.GuardCFFunctionTable.value = readDataInstance.readQword() # VA
        configDir.GuardCFFunctionCount.value = readDataInstance.readQword()
        configDir.GuardFlags.value = readDataInstance.readQword()
        return configDir