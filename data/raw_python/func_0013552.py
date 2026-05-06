def addDataset(self):
        """
        Adds a new dataset into this repo.
        """
        self._openRepo()
        dataset = datasets.Dataset(self._args.datasetName)
        dataset.setDescription(self._args.description)
        dataset.setAttributes(json.loads(self._args.attributes))
        self._updateRepo(self._repo.insertDataset, dataset)