def _getPrefixURL(self, url):
        """
        Given a url return namespace prefix.
        Leverages prefixes already in graph namespace
        Ex.  "http://www.drugbank.ca/drugs/DDD"
            -> "http://www.drugbank.ca/drugs/"
        """
        for prefix, namespace in self._rdfGraph.namespaces():
            if namespace.toPython() in url:
                return namespace