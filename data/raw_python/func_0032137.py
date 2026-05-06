def base_uri(self):
        """ Resolution base for JSON schema. Also used as the default
        graph ID for RDF. """
        if self._base_uri is None:
            if self._resolver is not None:
                self._base_uri = self.resolver.resolution_scope
            else:
                self._base_uri = 'http://pudo.github.io/jsongraph'
        return URIRef(self._base_uri)