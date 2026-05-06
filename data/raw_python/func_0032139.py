def store(self):
        """ Backend storage for RDF data. Either an in-memory store, or an
        external triple store controlled via SPARQL. """
        if self._store is None:
            config = self.config.get('store', {})
            if 'query' in config and 'update' in config:
                self._store = sparql_store(config.get('query'),
                                           config.get('update'))
            else:
                self._store = plugin.get('IOMemory', Store)()
            log.debug('Created store: %r', self._store)
        return self._store