def getSKOSDirectSubs(self, aURI):
        """
        2015-08-19: currenlty not used, inferred from above
        """
        aURI = aURI
        qres = self.rdfgraph.query(
              """SELECT DISTINCT ?x
                 WHERE {
                         {
                             { ?x skos:broader <%s> }
                             UNION
                             { <%s> skos:narrower ?s }
                         }
                     FILTER (!isBlank(?x))
                 }
                 """ % (aURI, aURI))
        return list(qres)