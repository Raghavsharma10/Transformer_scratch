def runGetRnaQuantificationSet(self, id_):
        """
        Runs a getRnaQuantificationSet request for the specified ID.
        """
        compoundId = datamodel.RnaQuantificationSetCompoundId.parse(id_)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        rnaQuantificationSet = dataset.getRnaQuantificationSet(id_)
        return self.runGetRequest(rnaQuantificationSet)