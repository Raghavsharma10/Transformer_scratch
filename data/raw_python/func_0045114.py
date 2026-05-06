def bestLabel(self, prefLanguage="en", qname_allowed=True, quotes=True):
        """
        facility for extrating the best available label for an entity

        ..This checks RFDS.label, SKOS.prefLabel and finally the qname local component
        """

        test = self.getValuesForProperty(rdflib.RDFS.label)
        out = ""

        if test:
            out = firstEnglishStringInList(test)
        else:
            test = self.getValuesForProperty(rdflib.namespace.SKOS.prefLabel)
            if test:
                out = firstEnglishStringInList(test)
            else:
                if qname_allowed:
                    out = self.locale

        if quotes and out:
            return addQuotes(out)
        else:
            return out