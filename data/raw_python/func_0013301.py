def toProtocolElement(self):
        """
        Converts this VariantAnnotationSet into its GA4GH protocol equivalent.
        """
        protocolElement = protocol.VariantAnnotationSet()
        protocolElement.id = self.getId()
        protocolElement.variant_set_id = self._variantSet.getId()
        protocolElement.name = self.getLocalId()
        protocolElement.analysis.CopyFrom(self.getAnalysis())
        self.serializeAttributes(protocolElement)
        return protocolElement