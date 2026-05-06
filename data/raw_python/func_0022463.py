def find_similar(self, doc, min_score=0.0, max_results=100):
        """
        Find `max_results` most similar articles in the index, each having similarity
        score of at least `min_score`. The resulting list may be shorter than `max_results`,
        in case there are not enough matching documents.

        `doc` is either a string (=document id, previously indexed) or a
        dict containing a 'tokens' key. These tokens are processed to produce a
        vector, which is then used as a query against the index.

        The similar documents are returned in decreasing similarity order, as
        `(doc_id, similarity_score, doc_payload)` 3-tuples. The payload returned
        is identical to what was supplied for this document during indexing.

        """
        logger.debug("received query call with %r" % doc)
        if self.is_locked():
            msg = "cannot query while the server is being updated"
            logger.error(msg)
            raise RuntimeError(msg)
        sims_opt, sims_fresh = None, None
        for index in [self.fresh_index, self.opt_index]:
            if index is not None:
                index.topsims = max_results
        if isinstance(doc, basestring):
            # query by direct document id
            docid = doc
            if self.opt_index is not None and docid in self.opt_index:
                sims_opt = self.opt_index.sims_by_id(docid)
                if self.fresh_index is not None:
                    vec = self.opt_index.vec_by_id(docid)
                    sims_fresh = self.fresh_index.sims_by_vec(vec, normalize=False)
            elif self.fresh_index is not None and docid in self.fresh_index:
                sims_fresh = self.fresh_index.sims_by_id(docid)
                if self.opt_index is not None:
                    vec = self.fresh_index.vec_by_id(docid)
                    sims_opt = self.opt_index.sims_by_vec(vec, normalize=False)
            else:
                raise ValueError("document %r not in index" % docid)
        else:
            if 'topics' in doc:
                # user supplied vector directly => use that
                vec = gensim.matutils.any2sparse(doc['topics'])
            else:
                # query by an arbitrary text (=tokens) inside doc['tokens']
                vec = self.model.doc2vec(doc) # convert document (text) to vector
            if self.opt_index is not None:
                sims_opt = self.opt_index.sims_by_vec(vec)
            if self.fresh_index is not None:
                sims_fresh = self.fresh_index.sims_by_vec(vec)

        merged = merge_sims(sims_opt, sims_fresh)
        logger.debug("got %s raw similars, pruning with max_results=%s, min_score=%s" %
            (len(merged), max_results, min_score))
        result = []
        for docid, score in merged:
            if score < min_score or 0 < max_results <= len(result):
                break
            result.append((docid, float(score), self.payload.get(docid, None)))
        return result