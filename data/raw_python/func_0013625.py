def _getIdentifier(self, url):
        """
        Given a url identifier return identifier portion
        Leverages prefixes already in graph namespace
        Returns None if no match
        Ex.  "http://www.drugbank.ca/drugs/DB01268" -> "DB01268"
        """
        for prefix, namespace in self._rdfGraph.namespaces():
            if namespace in url:
                return url.replace(namespace, '')