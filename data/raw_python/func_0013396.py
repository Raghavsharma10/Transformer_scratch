def getGaTermByName(self, name):
        """
        Returns a GA4GH OntologyTerm object by name.

        :param name: name of the ontology term, ex. "gene".
        :return: GA4GH OntologyTerm object.
        """
        # TODO what is the correct value when we have no mapping??
        termIds = self.getTermIds(name)
        if len(termIds) == 0:
            termId = ""
            # TODO add logging for missed term translation.
        else:
            # TODO what is the correct behaviour here when we have multiple
            # IDs matching a given name?
            termId = termIds[0]
        term = protocol.OntologyTerm()
        term.term = name
        term.term_id = termId
        return term