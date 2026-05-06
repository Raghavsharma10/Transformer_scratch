def addOntology(self, ontology):
        """
        Add an ontology map to this data repository.
        """
        self._ontologyNameMap[ontology.getName()] = ontology
        self._ontologyIdMap[ontology.getId()] = ontology
        self._ontologyIds.append(ontology.getId())