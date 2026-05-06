def standardize_genes(self, inplace=False):
        """Standardize the expression of each gene (row)."""
        matrix = self.center_genes(inplace=inplace)
        matrix.X[:,:] = matrix.X / \
            np.tile(np.std(matrix.X, axis=1, ddof=1), (matrix.n, 1)).T
        return matrix