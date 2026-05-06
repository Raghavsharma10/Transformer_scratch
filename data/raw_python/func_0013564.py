def addContinuousSet(self):
        """
        Adds a new continuous set into this repo
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        filePath = self._getFilePath(self._args.filePath,
                                     self._args.relativePath)
        name = getNameFromPath(self._args.filePath)
        continuousSet = continuous.FileContinuousSet(dataset, name)
        referenceSetName = self._args.referenceSetName
        if referenceSetName is None:
            raise exceptions.RepoManagerException(
                "A reference set name must be provided")
        referenceSet = self._repo.getReferenceSetByName(referenceSetName)
        continuousSet.setReferenceSet(referenceSet)
        continuousSet.populateFromFile(filePath)
        self._updateRepo(self._repo.insertContinuousSet, continuousSet)