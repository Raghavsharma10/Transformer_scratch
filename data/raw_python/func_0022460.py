def index(self, corpus=None, clear_buffer=True):
        """
        Permanently index all documents previously added via `buffer`, or
        directly index documents from `corpus`, if specified.

        The indexing model must already exist (see `train`) before this function
        is called.
        """
        if not self.model:
            msg = 'must initialize model for %s before indexing documents' % self.basename
            logger.error(msg)
            raise AttributeError(msg)

        if corpus is not None:
            # use the supplied corpus only (erase existing buffer, if any)
            self.flush(clear_buffer=True)
            self.buffer(corpus)

        if not self.fresh_docs:
            msg = "index called but no indexing corpus specified for %s" % self
            logger.error(msg)
            raise ValueError(msg)

        if not self.fresh_index:
            logger.info("starting a new fresh index for %s" % self)
            self.fresh_index = SimIndex(self.location('index_fresh'), self.model.num_features)
        self.fresh_index.index_documents(self.fresh_docs, self.model)
        if self.opt_index is not None:
            self.opt_index.delete(self.fresh_docs.keys())
        logger.info("storing document payloads")
        for docid in self.fresh_docs:
            payload = self.fresh_docs[docid].get('payload', None)
            if payload is None:
                # HACK: exit on first doc without a payload (=assume all docs have payload, or none does)
                break
            self.payload[docid] = payload
        self.flush(save_index=True, clear_buffer=clear_buffer)