def toProtocolElement(self):
        """
        Converts this rnaQuant into its GA4GH protocol equivalent.
        """
        protocolElement = protocol.RnaQuantification()
        protocolElement.id = self.getId()
        protocolElement.name = self._name
        protocolElement.description = self._description
        protocolElement.read_group_ids.extend(self._readGroupIds)
        protocolElement.programs.extend(self._programs)
        protocolElement.biosample_id = self._biosampleId
        protocolElement.feature_set_ids.extend(self._featureSetIds)
        protocolElement.rna_quantification_set_id = \
            self._parentContainer.getId()
        self.serializeAttributes(protocolElement)
        return protocolElement