def addBiosample(self):
        """
        Adds a new biosample into this repo
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        biosample = bio_metadata.Biosample(
            dataset, self._args.biosampleName)
        biosample.populateFromJson(self._args.biosample)
        self._updateRepo(self._repo.insertBiosample, biosample)