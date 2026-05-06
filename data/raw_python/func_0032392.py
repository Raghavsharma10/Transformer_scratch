def addSource(self, itemSource):
        """
        Add the given L{IBatchProcessor} as a source of input for this indexer.
        """
        _IndexerInputSource(store=self.store, indexer=self, source=itemSource)
        itemSource.addReliableListener(self, style=iaxiom.REMOTE)