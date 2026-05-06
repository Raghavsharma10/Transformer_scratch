def removeFeatureSet(self):
        """
        Removes a feature set from this repo
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        featureSet = dataset.getFeatureSetByName(self._args.featureSetName)

        def func():
            self._updateRepo(self._repo.removeFeatureSet, featureSet)
        self._confirmDelete("FeatureSet", featureSet.getLocalId(), func)