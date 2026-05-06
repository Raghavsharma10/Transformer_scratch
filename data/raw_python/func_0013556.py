def addPhenotypeAssociationSet(self):
        """
        Adds a new phenotype association set to this repo.
        """
        self._openRepo()
        name = self._args.name
        if name is None:
            name = getNameFromPath(self._args.dirPath)
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        phenotypeAssociationSet = \
            genotype_phenotype.RdfPhenotypeAssociationSet(
                dataset, name, self._args.dirPath)
        phenotypeAssociationSet.setAttributes(
            json.loads(self._args.attributes))
        self._updateRepo(
            self._repo.insertPhenotypeAssociationSet,
            phenotypeAssociationSet)