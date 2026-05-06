def delete(self, docids):
        """Delete documents (specified by their ids) from the index."""
        logger.debug("deleting %i documents from %s" % (len(docids), self))
        deleted = 0
        for docid in docids:
            try:
                del self.id2pos[docid]
                deleted += 1
                del self.id2sims[docid]
            except:
                pass
        self.id2sims.sync()
        if deleted:
            logger.info("deleted %i documents from %s" % (deleted, self))
        self.update_mappings()