def toProtocolElement(self):
        """
        Returns the GA4GH protocol representation of this ReadGroup.
        """
        # TODO this is very incomplete, but we don't have the
        # implementation to fill out the rest of the fields currently
        readGroup = protocol.ReadGroup()
        readGroup.id = self.getId()
        readGroup.created = self._creationTime
        readGroup.updated = self._updateTime
        dataset = self.getParentContainer().getParentContainer()
        readGroup.dataset_id = dataset.getId()
        readGroup.name = self.getLocalId()
        readGroup.predicted_insert_size = pb.int(self.getPredictedInsertSize())
        referenceSet = self._parentContainer.getReferenceSet()
        readGroup.sample_name = pb.string(self.getSampleName())
        readGroup.biosample_id = pb.string(self.getBiosampleId())
        if referenceSet is not None:
            readGroup.reference_set_id = referenceSet.getId()
        readGroup.stats.CopyFrom(self.getStats())
        readGroup.programs.extend(self.getPrograms())
        readGroup.description = pb.string(self.getDescription())
        readGroup.experiment.CopyFrom(self.getExperiment())
        self.serializeAttributes(readGroup)
        return readGroup