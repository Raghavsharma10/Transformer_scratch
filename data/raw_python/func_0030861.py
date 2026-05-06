def sample_correlations(self):
        """Returns an `ExpMatrix` containing all pairwise sample correlations.

        Returns
        -------
        `ExpMatrix`
            The sample correlation matrix.

        """
        C = np.corrcoef(self.X.T)
        corr_matrix = ExpMatrix(genes=self.samples, samples=self.samples, X=C)
        return corr_matrix