def toProtocolElement(self):
        """
        Converts this VariantSet into its GA4GH protocol equivalent.
        """
        protocolElement = protocol.VariantSet()
        protocolElement.id = self.getId()
        protocolElement.dataset_id = self.getParentContainer().getId()
        protocolElement.reference_set_id = self._referenceSet.getId()
        protocolElement.metadata.extend(self.getMetadata())
        protocolElement.dataset_id = self.getParentContainer().getId()
        protocolElement.reference_set_id = self._referenceSet.getId()
        protocolElement.name = self.getLocalId()
        self.serializeAttributes(protocolElement)
        return protocolElement