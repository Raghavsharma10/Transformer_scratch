def _parseExportDirectory(self, rva, size, magic = consts.PE32):
        """
        Parses the C{IMAGE_EXPORT_DIRECTORY} directory.
        
        @type rva: int 
        @param rva: The RVA where the C{IMAGE_EXPORT_DIRECTORY} directory starts.
        
        @type size: int
        @param size: The size of the C{IMAGE_EXPORT_DIRECTORY} directory.
        
        @type magic: int
        @param magic: (Optional) The type of PE. This value could be L{consts.PE32} or L{consts.PE64}.
        
        @rtype: L{ImageExportTable}
        @return: A new L{ImageExportTable} object.
        """
        data = self.getDataAtRva(rva,  size)
        rd = utils.ReadData(data)
        
        iet = directories.ImageExportTable.parse(rd)
        
        auxFunctionRvaArray = list()
        
        numberOfNames = iet.numberOfNames.value
        addressOfNames = iet.addressOfNames.value
        addressOfNameOrdinals = iet.addressOfNameOrdinals.value
        addressOfFunctions = iet.addressOfFunctions.value
        
        # populate the auxFunctionRvaArray
        for i in xrange(iet.numberOfFunctions.value):
            auxFunctionRvaArray.append(self.getDwordAtRva(addressOfFunctions).value)
            addressOfFunctions += datatypes.DWORD().sizeof()
            
        for i in xrange(numberOfNames):
            
            nameRva = self.getDwordAtRva(addressOfNames).value
            nameOrdinal = self.getWordAtRva(addressOfNameOrdinals).value
            exportName = self.readStringAtRva(nameRva).value
            
            entry = directories.ExportTableEntry()
            
            ordinal = nameOrdinal + iet.base.value
            #print "Ordinal value: %d" % ordinal
            entry.ordinal.value = ordinal
            
            entry.nameOrdinal.vaue = nameOrdinal
            entry.nameRva.value = nameRva
            entry.name.value = exportName
            entry.functionRva.value = auxFunctionRvaArray[nameOrdinal]
            
            iet.exportTable.append(entry)
            
            addressOfNames += datatypes.DWORD().sizeof()
            addressOfNameOrdinals += datatypes.WORD().sizeof()
        
        #print "export table length: %d" % len(iet.exportTable)
        
        #print "auxFunctionRvaArray: %r" % auxFunctionRvaArray
        for i in xrange(iet.numberOfFunctions.value):
            #print "auxFunctionRvaArray[%d]: %x" % (i,  auxFunctionRvaArray[i])
            if auxFunctionRvaArray[i] != iet.exportTable[i].functionRva.value:
                entry = directories.ExportTableEntry()
                
                entry.functionRva.value = auxFunctionRvaArray[i]
                entry.ordinal.value = iet.base.value + i
                
                iet.exportTable.append(entry)
        
        #print "export table length: %d" % len(iet.exportTable)
        sorted(iet.exportTable, key=lambda entry:entry.ordinal)
        return iet