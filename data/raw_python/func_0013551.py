def addOntology(self):
        """
        Adds a new Ontology to this repo.
        """
        self._openRepo()
        name = self._args.name
        filePath = self._getFilePath(self._args.filePath,
                                     self._args.relativePath)
        if name is None:
            name = getNameFromPath(filePath)
        ontology = ontologies.Ontology(name)
        ontology.populateFromFile(filePath)
        self._updateRepo(self._repo.insertOntology, ontology)