def _fixPe(self):
        """
        Fixes the necessary fields in the PE file instance in order to create a valid PE32. i.e. SizeOfImage.
        """
        sizeOfImage = 0
        for sh in self.sectionHeaders:
            sizeOfImage += sh.misc
        self.ntHeaders.optionaHeader.sizeoOfImage.value = self._sectionAlignment(sizeOfImage + 0x1000)