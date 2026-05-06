def entityTriples(self, aURI):
        """ Builds all triples for an entity
        Note: if a triple object is a blank node (=a nested definition)
        we try to extract all relevant data recursively (does not work with
        sparql endpoins)
        2015-10-18: updated
        """

        aURI = aURI
        qres = self.rdfgraph.query(
              """CONSTRUCT {<%s> ?y ?z }
                 WHERE {
                     { <%s> ?y ?z }
                 }
                 """ % (aURI, aURI ))
        lres = list(qres)

        def recurse(triples_list):
            """ uses the rdflib <triples> method to pull out all blank nodes info"""
            out = []
            for tripl in triples_list:
                if isBlankNode(tripl[2]):
                    # print "blank node", str(tripl[2])
                    temp = [x for x in self.rdfgraph.triples((tripl[2], None, None))]
                    out += temp + recurse(temp)
                else:
                    pass
            return out

        try:
            return lres + recurse(lres)
        except:
            printDebug("Error extracting blank nodes info", "important")
            return lres