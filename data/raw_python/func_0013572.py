def removeOntology(self):
        """
        Removes an ontology from the repo.
        """
        self._openRepo()
        ontology = self._repo.getOntologyByName(self._args.ontologyName)

        def func():
            self._updateRepo(self._repo.removeOntology, ontology)
        self._confirmDelete("Ontology", ontology.getName(), func)