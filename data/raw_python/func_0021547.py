def parse(readDataInstance):
        """
        Returns a new L{ImageBoundImportDescriptor} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object containing the data to create a new L{ImageBoundImportDescriptor} object.
        
        @rtype: L{ImageBoundImportDescriptor}
        @return: A new {ImageBoundImportDescriptor} object.
        """
        ibd = ImageBoundImportDescriptor()
        
        entryData = readDataInstance.read(consts.SIZEOF_IMAGE_BOUND_IMPORT_ENTRY32)
        readDataInstance.offset = 0
        while not utils.allZero(entryData):
            prevOffset = readDataInstance.offset
            
            boundEntry = ImageBoundImportDescriptorEntry.parse(readDataInstance)
            
            # if the parsed entry has numberOfModuleForwarderRefs we must adjust the value in the readDataInstance.offset field
            # in order to point after the last ImageBoundForwarderRefEntry.
            if boundEntry.numberOfModuleForwarderRefs.value:
                readDataInstance.offset = prevOffset + (consts.SIZEOF_IMAGE_BOUND_FORWARDER_REF_ENTRY32 * boundEntry.numberOfModuleForwarderRefs.value)
            else:
                readDataInstance.offset = prevOffset
            
            ibd.append(boundEntry)
            entryData = readDataInstance.read(consts.SIZEOF_IMAGE_BOUND_IMPORT_ENTRY32)
            
        return ibd