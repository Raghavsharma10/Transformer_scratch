def removeContinuousSet(self):
        """
        Removes a continuous set from this repo
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        continuousSet = dataset.getContinuousSetByName(
                            self._args.continuousSetName)

        def func():
            self._updateRepo(self._repo.removeContinuousSet, continuousSet)
        self._confirmDelete("ContinuousSet", continuousSet.getLocalId(), func)