def runGetBiosample(self, id_):
        """
        Runs a getBiosample request for the specified ID.
        """
        compoundId = datamodel.BiosampleCompoundId.parse(id_)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        biosample = dataset.getBiosample(id_)
        return self.runGetRequest(biosample)