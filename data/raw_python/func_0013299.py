def getMetadataId(self, metadata):
        """
        Returns the id of a metadata
        """
        return str(datamodel.VariantSetMetadataCompoundId(
            self.getCompoundId(), 'metadata:' + metadata.key))