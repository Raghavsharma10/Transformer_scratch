def filter_genes(self, gene_names : Iterable[str]):
        """Filter the expression matrix against a set of genes.

        Parameters
        ----------
        gene_names: list of str
            The genome to filter the genes against.

        Returns
        -------
        ExpMatrix
            The filtered expression matrix.
        """

        filt = self.loc[self.index & gene_names]
        return filt