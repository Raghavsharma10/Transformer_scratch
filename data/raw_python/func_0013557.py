def removePhenotypeAssociationSet(self):
        """
        Removes a phenotype association set from the repo
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        phenotypeAssociationSet = dataset.getPhenotypeAssociationSetByName(
            self._args.name)

        def func():
            self._updateRepo(
                self._repo.removePhenotypeAssociationSet,
                phenotypeAssociationSet)
        self._confirmDelete(
            "PhenotypeAssociationSet",
            phenotypeAssociationSet.getLocalId(),
            func)