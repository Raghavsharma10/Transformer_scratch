def runGetRnaQuantification(self, id_):
        """
        Runs a getRnaQuantification request for the specified ID.
        """
        compoundId = datamodel.RnaQuantificationCompoundId.parse(id_)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        rnaQuantificationSet = dataset.getRnaQuantificationSet(
            compoundId.rna_quantification_set_id)
        rnaQuantification = rnaQuantificationSet.getRnaQuantification(id_)
        return self.runGetRequest(rnaQuantification)