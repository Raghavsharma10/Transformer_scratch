def serialize(self, format="turtle"):
        """ xml, n3, turtle, nt, pretty-xml, trix are built in"""
        if self.triples:
            if not self.rdfgraph:
                self._buildGraph()
            return self.rdfgraph.serialize(format=format)
        else:
            return None