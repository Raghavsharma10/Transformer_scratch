def bestDescription(self, prefLanguage="en"):
        """
        facility for extrating the best available description for an entity

        ..This checks RFDS.label, SKOS.prefLabel and finally the qname local component
        """

        test_preds = [rdflib.RDFS.comment, rdflib.namespace.DCTERMS.description, rdflib.namespace.DC.description,
                      rdflib.namespace.SKOS.definition]

        for pred in test_preds:
            test = self.getValuesForProperty(pred)
            if test:
                return addQuotes(firstEnglishStringInList(test))
        return ""