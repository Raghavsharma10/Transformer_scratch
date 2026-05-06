def buffer(self, documents):
        """
        Add a sequence of documents to be processed (indexed or trained on).

        Here, the documents are simply collected; real processing is done later,
        during the `self.index` or `self.train` calls.

        `buffer` can be called repeatedly; the result is the same as if it was
        called once, with a concatenation of all the partial document batches.
        The point is to save memory when sending large corpora over network: the
        entire `documents` must be serialized into RAM. See `utils.upload_chunked()`.

        A call to `flush()` clears this documents-to-be-processed buffer (`flush`
        is also implicitly called when you call `index()` and `train()`).
        """
        logger.info("adding documents to temporary buffer of %s" % (self))
        for doc in documents:
            docid = doc['id']
#            logger.debug("buffering document %r" % docid)
            if docid in self.fresh_docs:
                logger.warning("asked to re-add id %r; rewriting old value" % docid)
            self.fresh_docs[docid] = doc
        self.fresh_docs.sync()