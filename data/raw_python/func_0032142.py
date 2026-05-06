def context(self, identifier=None, meta=None):
        """ Get or create a context, with the given identifier and/or
        provenance meta data. A context can be used to add, update or delete
        objects in the store. """
        return Context(self, identifier=identifier, meta=meta)