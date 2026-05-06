def addRnaQuantification(self):
        """
        Adds an rnaQuantification into this repo
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        biosampleId = ""
        if self._args.biosampleName:
            biosample = dataset.getBiosampleByName(self._args.biosampleName)
            biosampleId = biosample.getId()
        if self._args.name is None:
            name = getNameFromPath(self._args.quantificationFilePath)
        else:
            name = self._args.name
        # TODO: programs not fully supported by GA4GH yet
        programs = ""
        featureType = "gene"
        if self._args.transcript:
            featureType = "transcript"
        rnaseq2ga.rnaseq2ga(
            self._args.quantificationFilePath, self._args.filePath, name,
            self._args.format, dataset=dataset, featureType=featureType,
            description=self._args.description, programs=programs,
            featureSetNames=self._args.featureSetNames,
            readGroupSetNames=self._args.readGroupSetName,
            biosampleId=biosampleId)