def addIndividual(self):
        """
        Adds a new individual into this repo
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        individual = bio_metadata.Individual(
            dataset, self._args.individualName)
        individual.populateFromJson(self._args.individual)
        self._updateRepo(self._repo.insertIndividual, individual)