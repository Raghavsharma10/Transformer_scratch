def _updateMetadata(self, variantFile):
        """
        Updates the metadata for his variant set based on the specified
        variant file
        """
        metadata = self._getMetadataFromVcf(variantFile)
        if self._metadata is None:
            self._metadata = metadata