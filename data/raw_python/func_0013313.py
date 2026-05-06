def convertSeqOntology(self, seqOntStr):
        """
        Splits a string of sequence ontology effects and creates
        an ontology term record for each, which are built into
        an array of return soTerms.
        :param seqOntStr:
        :return: [protocol.OntologyTerm]
        """
        return [
            self._ontology.getGaTermByName(soName)
            for soName in seqOntStr.split('&')]