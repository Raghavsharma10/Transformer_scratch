def _scan(self, verbose=False, hide_base_schemas=True):
        """
        scan a source of RDF triples
        build all the objects to deal with the ontology/ies pythonically

        """
        if verbose:
            printDebug("Scanning entities...", "green")
            printDebug("----------", "comment")

        self.__extractOntologies()
        if verbose: printDebug("Ontologies.........: %d" % len(self.ontologies), "comment")

        self.__extractClasses(hide_base_schemas)
        if verbose: printDebug("Classes............: %d" % len(self.classes), "comment")

        self.__extractProperties()
        if verbose: printDebug("Properties.........: %d" % len(self.properties), "comment")
        if verbose: printDebug("..annotation.......: %d" % len(self.annotationProperties), "comment")
        if verbose: printDebug("..datatype.........: %d" % len(self.datatypeProperties), "comment")
        if verbose: printDebug("..object...........: %d" % len(self.objectProperties), "comment")

        self.__extractSkosConcepts()
        if verbose: printDebug("Concepts (SKOS)....: %d" % len(self.skosConcepts), "comment")

        self.__computeTopLayer()

        self.__computeInferredProperties()

        if verbose: printDebug("----------", "comment")