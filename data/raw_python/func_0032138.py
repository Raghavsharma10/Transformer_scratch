def resolver(self):
        """ Resolver for JSON Schema references. This can be based around a
        file or HTTP-based resolution base URI. """
        if self._resolver is None:
            self._resolver = RefResolver(self.base_uri, {})
        # if self.base_uri not in self._resolver.store:
        #    self._resolver.store[self.base_uri] = self.config
        return self._resolver