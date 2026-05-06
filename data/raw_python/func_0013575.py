def addRnaQuantificationSet(self):
        """
        Adds an rnaQuantificationSet into this repo
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        if self._args.name is None:
            name = getNameFromPath(self._args.filePath)
        else:
            name = self._args.name
        rnaQuantificationSet = rna_quantification.SqliteRnaQuantificationSet(
            dataset, name)
        referenceSetName = self._args.referenceSetName
        if referenceSetName is None:
            raise exceptions.RepoManagerException(
                "A reference set name must be provided")
        referenceSet = self._repo.getReferenceSetByName(referenceSetName)
        rnaQuantificationSet.setReferenceSet(referenceSet)
        rnaQuantificationSet.populateFromFile(self._args.filePath)
        rnaQuantificationSet.setAttributes(json.loads(self._args.attributes))
        self._updateRepo(
            self._repo.insertRnaQuantificationSet, rnaQuantificationSet)