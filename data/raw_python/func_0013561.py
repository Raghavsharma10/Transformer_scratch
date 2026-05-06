def removeDataset(self):
        """
        Removes a dataset from the repo.
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)

        def func():
            self._updateRepo(self._repo.removeDataset, dataset)
        self._confirmDelete("Dataset", dataset.getLocalId(), func)