def runGetFeature(self, id_):
        """
        Returns JSON string of the feature object corresponding to
        the feature compoundID passed in.
        """
        compoundId = datamodel.FeatureCompoundId.parse(id_)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        featureSet = dataset.getFeatureSet(compoundId.feature_set_id)
        gaFeature = featureSet.getFeature(compoundId)
        jsonString = protocol.toJson(gaFeature)
        return jsonString