def sims_by_vec(self, vec, normalize=None):
        """
        Find the most similar documents to a given vector (=already processed document).
        """
        if normalize is None:
            normalize = self.qindex.normalize
        norm, self.qindex.normalize = self.qindex.normalize, normalize # store old value
        self.qindex.num_best = self.topsims
        sims = self.qindex[vec]
        self.qindex.normalize = norm # restore old value of qindex.normalize
        return self.sims2scores(sims)