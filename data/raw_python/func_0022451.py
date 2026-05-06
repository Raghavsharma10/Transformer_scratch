def sims_by_id(self, docid):
        """Find the most similar documents to the (already indexed) document with `docid`."""
        result = self.id2sims.get(docid, None)
        if result is None:
            self.qindex.num_best = self.topsims
            sims = self.qindex.similarity_by_id(self.id2pos[docid])
            result = self.sims2scores(sims)
        return result