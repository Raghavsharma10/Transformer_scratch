def sparql(self, stringa):
        """ wrapper around a sparql query """
        qres = self.rdfgraph.query(stringa)
        return list(qres)