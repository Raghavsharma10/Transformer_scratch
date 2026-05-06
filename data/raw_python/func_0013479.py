def runGetReadGroup(self, id_):
        """
        Returns a read group with the given id_
        """
        compoundId = datamodel.ReadGroupCompoundId.parse(id_)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        readGroupSet = dataset.getReadGroupSet(compoundId.read_group_set_id)
        readGroup = readGroupSet.getReadGroup(id_)
        return self.runGetRequest(readGroup)