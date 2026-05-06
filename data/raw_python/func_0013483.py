def runGetFeatureSet(self, id_):
        """
        Runs a getFeatureSet request for the specified ID.
        """
        compoundId = datamodel.FeatureSetCompoundId.parse(id_)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        featureSet = dataset.getFeatureSet(id_)
        return self.runGetRequest(featureSet)