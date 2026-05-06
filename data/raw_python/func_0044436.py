def setQuery(self, query):
        """
        Set the SPARQL query text and set the VIVO custom
        authentication parameters.

        Set here because this is called immediately before
        any query is sent to the triple store.
        """
        self.queryType = self._parseQueryType(query)
        self.queryString = self.injectPrefixes(query)
        self.addParameter('email', self.email)
        self.addParameter('password', self.password)