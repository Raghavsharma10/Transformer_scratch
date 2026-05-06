def _flush(self):
        """
        Deal with pending result-affecting things.

        This should always be called before issuing a search.
        """
        remove = self.store.query(_RemoveDocument)
        documentIdentifiers = list(remove.getColumn("documentIdentifier"))
        if VERBOSE:
            log.msg("%s/%d removing %r" % (self.store, self.storeID, documentIdentifiers))
        reader = self.openReadIndex()
        map(reader.remove, documentIdentifiers)
        reader.close()
        remove.deleteFromStore()