def register(self, alias, uri):
        """ Register a new schema URI under a given name. """
        # TODO: do we want to constrain the valid alias names.
        if isinstance(uri, dict):
            id = uri.get('id', alias)
            self.resolver.store[id] = uri
            uri = id
        self.aliases[alias] = uri