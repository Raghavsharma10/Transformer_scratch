def addReferenceSet(self):
        """
        Adds a new reference set into this repo.
        """
        self._openRepo()
        name = self._args.name
        filePath = self._getFilePath(self._args.filePath,
                                     self._args.relativePath)
        if name is None:
            name = getNameFromPath(self._args.filePath)
        referenceSet = references.HtslibReferenceSet(name)
        referenceSet.populateFromFile(filePath)
        referenceSet.setDescription(self._args.description)
        if self._args.species is not None:
            referenceSet.setSpeciesFromJson(self._args.species)
        referenceSet.setIsDerived(self._args.isDerived)
        referenceSet.setAssemblyId(self._args.assemblyId)
        referenceSet.setAttributes(json.loads(self._args.attributes))
        sourceAccessions = []
        if self._args.sourceAccessions is not None:
            sourceAccessions = self._args.sourceAccessions.split(",")
        referenceSet.setSourceAccessions(sourceAccessions)
        referenceSet.setSourceUri(self._args.sourceUri)
        self._updateRepo(self._repo.insertReferenceSet, referenceSet)