def toProtocolElement(self):
        """
        Returns the representation of this ContinuousSet as the corresponding
        ProtocolElement.
        """
        gaContinuousSet = protocol.ContinuousSet()
        gaContinuousSet.id = self.getId()
        gaContinuousSet.dataset_id = self.getParentContainer().getId()
        gaContinuousSet.reference_set_id = pb.string(
                                            self._referenceSet.getId())
        gaContinuousSet.name = self._name
        gaContinuousSet.source_uri = self._sourceUri
        attributes = self.getAttributes()
        for key in attributes:
            gaContinuousSet.attributes.attr[key] \
                .values.extend(protocol.encodeValue(attributes[key]))
        return gaContinuousSet