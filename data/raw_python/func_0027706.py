def _schemaPrepareInsert(self, store):
        """
        Prepare each attribute in my schema for insertion into a given store,
        either by upgrade or by creation.  This makes sure all references point
        to this store and all relative paths point to this store's files
        directory.
        """
        for name, atr in self.getSchema():
            atr.prepareInsert(self, store)