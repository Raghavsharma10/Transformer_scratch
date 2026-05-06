def _internalParse(self, readDataInstance):
        """
        Populates the attributes of the L{PE} object. 
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} instance with the data of a PE file.
        """
        self.dosHeader = DosHeader.parse(readDataInstance)
        
        self.dosStub = readDataInstance.read(self.dosHeader.e_lfanew.value - readDataInstance.offset)
        self.ntHeaders = NtHeaders.parse(readDataInstance)
        
        if self.ntHeaders.optionalHeader.magic.value == consts.PE32:
            self.PE_TYPE = consts.PE32
        elif self.ntHeaders.optionalHeader.magic.value == consts.PE64:
            self.PE_TYPE = consts.PE64
            readDataInstance.setOffset(readDataInstance.tell() - OptionalHeader().sizeof())
            self.ntHeaders.optionalHeader = OptionalHeader64.parse(readDataInstance)
            
        self.sectionHeaders = SectionHeaders.parse(readDataInstance,  self.ntHeaders.fileHeader.numberOfSections.value)

        # as padding is possible between the last section header and the beginning of the first section
        # we must adjust the offset in readDataInstance to point to the first byte of the first section.
        readDataInstance.setOffset(self.sectionHeaders[0].pointerToRawData.value)
        
        self.sections = Sections.parse(readDataInstance,  self.sectionHeaders)
        
        self.overlay = self._getOverlay(readDataInstance,  self.sectionHeaders)
        self.signature = self._getSignature(readDataInstance,  self.ntHeaders.optionalHeader.dataDirectory)
        
        if not self._fastLoad:
            self._parseDirectories(self.ntHeaders.optionalHeader.dataDirectory, self.PE_TYPE)