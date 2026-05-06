def parse(readDataInstance):
        """
        Returns a new L{ImageBoundImportDescriptorEntry} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object containing data to create a new L{ImageBoundImportDescriptorEntry}.
        
        @rtype: L{ImageBoundImportDescriptorEntry}
        @return: A new {ImageBoundImportDescriptorEntry} object.
        """
        boundEntry = ImageBoundImportDescriptorEntry()
        boundEntry.timeDateStamp.value = readDataInstance.readDword()
        boundEntry.offsetModuleName.value = readDataInstance.readWord()
        boundEntry.numberOfModuleForwarderRefs.value = readDataInstance.readWord()
        
        numberOfForwarderRefsEntries = boundEntry.numberOfModuleForwarderRefs .value
        if numberOfForwarderRefsEntries:
            bytesToRead = numberOfForwarderRefsEntries * ImageBoundForwarderRefEntry().sizeof()
            rd = utils.ReadData(readDataInstance.read(bytesToRead))
            boundEntry.forwarderRefsList = ImageBoundForwarderRef.parse(rd,  numberOfForwarderRefsEntries)
            
        return boundEntry