def toProtocolElement(self):
        """
        Returns the representation of this CallSet as the corresponding
        ProtocolElement.
        """
        variantSet = self.getParentContainer()
        gaCallSet = protocol.CallSet(
            biosample_id=self.getBiosampleId())
        if variantSet.getCreationTime():
            gaCallSet.created = variantSet.getCreationTime()
        if variantSet.getUpdatedTime():
            gaCallSet.updated = variantSet.getUpdatedTime()
        gaCallSet.id = self.getId()
        gaCallSet.name = self.getLocalId()
        gaCallSet.variant_set_ids.append(variantSet.getId())
        self.serializeAttributes(gaCallSet)
        return gaCallSet