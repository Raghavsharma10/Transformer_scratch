def toProtocolElement(self):
        """
        Converts this rnaQuant into its GA4GH protocol equivalent.
        """
        protocolElement = protocol.RnaQuantificationSet()
        protocolElement.id = self.getId()
        protocolElement.dataset_id = self._parentContainer.getId()
        protocolElement.name = self._name
        self.serializeAttributes(protocolElement)
        return protocolElement