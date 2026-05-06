def removeRnaQuantificationSet(self):
        """
        Removes an rnaQuantificationSet from this repo
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        rnaQuantSet = dataset.getRnaQuantificationSetByName(
            self._args.rnaQuantificationSetName)

        def func():
            self._updateRepo(self._repo.removeRnaQuantificationSet,
                             rnaQuantSet)
        self._confirmDelete(
            "RnaQuantificationSet", rnaQuantSet.getLocalId(), func)