def _addDataFile(self, filename):
        """
        Given a filename, add it to the graph
        """
        if filename.endswith('.ttl'):
            self._rdfGraph.parse(filename, format='n3')
        else:
            self._rdfGraph.parse(filename, format='xml')