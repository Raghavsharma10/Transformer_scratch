def addSection(self, data, name =".pype32\x00", flags = 0x60000000):
        """
        Adds a new section to the existing L{PE} instance.
        
        @type data: str
        @param data: The data to be added in the new section.
        
        @type name: str
        @param name: (Optional) The name for the new section.
        
        @type flags: int
        @param flags: (Optional) The attributes for the new section.
        """
        fa = self.ntHeaders.optionalHeader.fileAlignment.value
        sa = self.ntHeaders.optionalHeader.sectionAlignment.value

        padding = "\xcc" * (fa - len(data))
        sh = SectionHeader()
        
        if len(self.sectionHeaders):
            # get the va, vz, ra and rz of the last section in the array of section headers
            vaLastSection = self.sectionHeaders[-1].virtualAddress.value
            sizeLastSection = self.sectionHeaders[-1].misc.value
            pointerToRawDataLastSection = self.sectionHeaders[-1].pointerToRawData.value
            sizeOfRawDataLastSection = self.sectionHeaders[-1].sizeOfRawData.value
            
            sh.virtualAddress.value = self._adjustSectionAlignment(vaLastSection + sizeLastSection,  fa, sa)
            sh.pointerToRawData.value = self._adjustFileAlignment(pointerToRawDataLastSection + sizeOfRawDataLastSection,  fa)

        sh.misc.value = self._adjustSectionAlignment(len(data),  fa,  sa) or consts.DEFAULT_PAGE_SIZE
        sh.sizeOfRawData.value = self._adjustFileAlignment(len(data),  fa) or consts.DEFAULT_FILE_ALIGNMENT            
        sh.characteristics.value = flags
        sh.name.value = name
        
        self.sectionHeaders.append(sh)
        self.sections.append(data + padding)
        
        self.ntHeaders.fileHeader.numberOfSections.value += 1