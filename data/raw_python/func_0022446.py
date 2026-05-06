def update_ids(self, docids):
        """Update id->pos mapping with new document ids."""
        logger.info("updating %i id mappings" % len(docids))
        for docid in docids:
            if docid is not None:
                pos = self.id2pos.get(docid, None)
                if pos is not None:
                    logger.info("replacing existing document %r in %s" % (docid, self))
                    del self.pos2id[pos]
                self.id2pos[docid] = self.length
                try:
                    del self.id2sims[docid]
                except:
                    pass
            self.length += 1
        self.id2sims.sync()
        self.update_mappings()