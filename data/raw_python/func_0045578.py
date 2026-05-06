def getClassDirectSubs(self, aURI):
        """
        2015-06-03: currenlty not used, inferred from above
        """
        aURI = aURI
        qres = self.rdfgraph.query(
              """SELECT DISTINCT ?x
                 WHERE {
                     { ?x rdfs:subClassOf <%s> }
                     FILTER (!isBlank(?x))
                 }
                 """ % (aURI))
        return list(qres)