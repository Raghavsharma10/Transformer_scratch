def removeReadGroupSet(self):
        """
        Removes a readGroupSet from the repo.
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        readGroupSet = dataset.getReadGroupSetByName(
            self._args.readGroupSetName)

        def func():
            self._updateRepo(self._repo.removeReadGroupSet, readGroupSet)
        self._confirmDelete("ReadGroupSet", readGroupSet.getLocalId(), func)