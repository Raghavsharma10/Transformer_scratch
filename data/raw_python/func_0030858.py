def center_genes(self, use_median=False, inplace=False):
        """Center the expression of each gene (row)."""
        if use_median:
            X = self.X - \
                np.tile(np.median(self.X, axis=1), (self.n, 1)).T
        else:
            X = self.X - \
                np.tile(np.mean(self.X, axis=1), (self.n, 1)).T

        if inplace:
            self.X[:,:] = X
            matrix = self
        else:
            matrix = ExpMatrix(genes=self.genes, samples=self.samples,
                               X=X)
        return matrix