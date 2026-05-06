def removeIndividual(self):
        """
        Removes an individual from this repo
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        individual = dataset.getIndividualByName(self._args.individualName)

        def func():
            self._updateRepo(self._repo.removeIndividual, individual)
        self._confirmDelete("Individual", individual.getLocalId(), func)