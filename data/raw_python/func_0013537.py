def toProtocolElement(self):
        """
        Returns the GA4GH protocol representation of this ReadGroupSet.
        """
        readGroupSet = protocol.ReadGroupSet()
        readGroupSet.id = self.getId()
        readGroupSet.read_groups.extend(
            [readGroup.toProtocolElement()
             for readGroup in self.getReadGroups()]
        )
        readGroupSet.name = self.getLocalId()
        readGroupSet.dataset_id = self.getParentContainer().getId()
        readGroupSet.stats.CopyFrom(self.getStats())
        self.serializeAttributes(readGroupSet)
        return readGroupSet