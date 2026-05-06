def getClassAllSupers(self, aURI):
        """
        note: requires SPARQL 1.1
        2015-06-04: currenlty not used, inferred from above
        """
        aURI = aURI
        try:
            qres = self.rdfgraph.query(
                  """SELECT DISTINCT ?x
                     WHERE {
                         { <%s> rdfs:subClassOf+ ?x }
                         FILTER (!isBlank(?x))
                     }
                     """ % (aURI))
        except:
            printDebug("... warning: the 'getClassAllSupers' query failed (maybe missing SPARQL 1.1 support?)")
            qres = []
        return list(qres)