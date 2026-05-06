def removeBiosample(self):
        """
        Removes a biosample from this repo
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        biosample = dataset.getBiosampleByName(self._args.biosampleName)

        def func():
            self._updateRepo(self._repo.removeBiosample, biosample)
        self._confirmDelete("Biosample", biosample.getLocalId(), func)