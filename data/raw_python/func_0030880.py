def fold_enrichment(self):
        """Returns the fold enrichment of the gene set.

        Fold enrichment is defined as ratio between the observed and the
        expected number of gene set genes present.
        """
        expected = self.K * (self.n/float(self.N))
        return self.k / expected