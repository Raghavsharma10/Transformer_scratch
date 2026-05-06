def train(self, corpus=None, method='auto', clear_buffer=True, params=None):
        """
        Create an indexing model. Will overwrite the model if it already exists.
        All indexes become invalid, because documents in them use a now-obsolete
        representation.

        The model is trained on documents previously entered via `buffer`,
        or directly on `corpus`, if specified.
        """
        if corpus is not None:
            # use the supplied corpus only (erase existing buffer, if any)
            self.flush(clear_buffer=True)
            self.buffer(corpus)
        if not self.fresh_docs:
            msg = "train called but no training corpus specified for %s" % self
            logger.error(msg)
            raise ValueError(msg)
        if method == 'auto':
            numdocs = len(self.fresh_docs)
            if numdocs < 1000:
                logging.warning("too few training documents; using simple log-entropy model instead of latent semantic indexing")
                method = 'logentropy'
            else:
                method = 'lsi'
        if params is None:
            params = {}
        self.model = SimModel(self.fresh_docs, method=method, params=params)
        self.flush(save_model=True, clear_buffer=clear_buffer)