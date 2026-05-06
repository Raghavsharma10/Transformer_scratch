def runGetIndividual(self, id_):
        """
        Runs a getIndividual request for the specified ID.
        """
        compoundId = datamodel.BiosampleCompoundId.parse(id_)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        individual = dataset.getIndividual(id_)
        return self.runGetRequest(individual)