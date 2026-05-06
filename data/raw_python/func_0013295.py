def _updateCallSetIds(self, variantFile):
        """
        Updates the call set IDs based on the specified variant file.
        """
        if len(self._callSetIdMap) == 0:
            for sample in variantFile.header.samples:
                self.addCallSetFromName(sample)