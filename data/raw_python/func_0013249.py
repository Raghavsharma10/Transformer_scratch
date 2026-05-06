def toProtocolElement(self):
        """
        Returns the representation of this FeatureSet as the corresponding
        ProtocolElement.
        """
        gaFeatureSet = protocol.FeatureSet()
        gaFeatureSet.id = self.getId()
        gaFeatureSet.dataset_id = self.getParentContainer().getId()
        gaFeatureSet.reference_set_id = pb.string(self._referenceSet.getId())
        gaFeatureSet.name = self._name
        gaFeatureSet.source_uri = self._sourceUri
        attributes = self.getAttributes()
        for key in attributes:
            gaFeatureSet.attributes.attr[key] \
                .values.extend(protocol.encodeValue(attributes[key]))
        return gaFeatureSet