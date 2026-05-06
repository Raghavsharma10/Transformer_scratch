def parse(readDataInstance):
        """
        Returns a new L{NetDirectory} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{NetDirectory} object.
        
        @rtype: L{NetDirectory}
        @return: A new L{NetDirectory} object.
        """
        nd = NetDirectory()
        
        nd.cb.value = readDataInstance.readDword()
        nd.majorRuntimeVersion.value= readDataInstance.readWord()
        nd.minorRuntimeVersion.value = readDataInstance.readWord()
        
        nd.metaData.rva.value = readDataInstance.readDword()
        nd.metaData.size.value = readDataInstance.readDword()
        nd.metaData.name.value = "MetaData"
        
        nd.flags.value = readDataInstance.readDword()
        nd.entryPointToken.value = readDataInstance.readDword()
        
        nd.resources.rva.value = readDataInstance.readDword()
        nd.resources.size.value = readDataInstance.readDword()
        nd.resources.name.value = "Resources"
        
        nd.strongNameSignature.rva.value = readDataInstance.readDword()
        nd.strongNameSignature.size.value = readDataInstance.readDword()
        nd.strongNameSignature.name.value = "StrongNameSignature"
        
        nd.codeManagerTable.rva.value = readDataInstance.readDword()
        nd.codeManagerTable.size.value = readDataInstance.readDword()
        nd.codeManagerTable.name.value = "CodeManagerTable"
        
        nd.vTableFixups.rva.value = readDataInstance.readDword()
        nd.vTableFixups.size.value = readDataInstance.readDword()
        nd.vTableFixups.name.value = "VTableFixups"
        
        nd.exportAddressTableJumps.rva.value = readDataInstance.readDword()
        nd.exportAddressTableJumps.size.value = readDataInstance.readDword()
        nd.exportAddressTableJumps.name.value = "ExportAddressTableJumps"
        
        nd.managedNativeHeader.rva.value = readDataInstance.readDword()
        nd.managedNativeHeader.size.value = readDataInstance.readDword()
        nd.managedNativeHeader.name.value = "ManagedNativeHeader"
        
        return nd