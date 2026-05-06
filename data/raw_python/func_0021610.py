def fullLoad(self):
        """Parse all the directories in the PE file."""
        self._parseDirectories(self.ntHeaders.optionalHeader.dataDirectory, self.PE_TYPE)