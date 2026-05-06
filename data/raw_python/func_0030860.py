def filter_genes(self, gene_names : Iterable[str], inplace=False):
        """Filter the expression matrix against a _genome (set of genes).

        Parameters
        ----------
        gene_names: list of str
            The genome to filter the genes against.
        inplace: bool, optional
            Whether to perform the operation in-place.

        Returns
        -------
        ExpMatrix
            The filtered expression matrix.
        """

        return self.drop(set(self.genes) - set(gene_names),
                         inplace=inplace)