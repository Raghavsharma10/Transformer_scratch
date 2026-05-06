def runGetContinuousSet(self, id_):
        """
        Runs a getContinuousSet request for the specified ID.
        """
        compoundId = datamodel.ContinuousSetCompoundId.parse(id_)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        continuousSet = dataset.getContinuousSet(id_)
        return self.runGetRequest(continuousSet)