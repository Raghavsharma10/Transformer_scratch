def optimize(self):
        """
        Precompute top similarities for all indexed documents. This speeds up
        `find_similar` queries by id (but not queries by fulltext).

        Internally, documents are moved from a fresh index (=no precomputed similarities)
        to an optimized index (precomputed similarities). Similarity queries always
        query both indexes, so this split is transparent to clients.

        If you add documents later via `index`, they go to the fresh index again.
        To precompute top similarities for these new documents too, simply call
        `optimize` again.

        """
        if self.fresh_index is None:
            logger.warning("optimize called but there are no new documents")
            return # nothing to do!

        if self.opt_index is None:
            logger.info("starting a new optimized index for %s" % self)
            self.opt_index = SimIndex(self.location('index_opt'), self.model.num_features)

        self.opt_index.merge(self.fresh_index)
        self.fresh_index.terminate() # delete old files
        self.fresh_index = None
        self.flush(save_index=True)