def nt_yielder(self, graph, size):
        """
        Yield n sized ntriples for a given graph.
        Used in sending chunks of data to the VIVO
        SPARQL API.
        """
        for grp in self.make_batch(size, graph):
            tmpg = Graph()
            # Add statements as list to tmp graph
            tmpg += grp
            yield (len(tmpg), tmpg.serialize(format='nt'))