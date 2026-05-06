def read_tsv(cls, filepath_or_buffer: str, gene_table: ExpGeneTable = None,
                 encoding='UTF-8'):
        """Read expression profile from a tab-delimited text file.

        Parameters
        ----------
        path: str
            The path of the text file.
        gene_table: `ExpGeneTable` object, optional
            The set of valid genes. If given, the genes in the text file will
            be filtered against this set of genes. (None)
        encoding: str, optional
            The file encoding. ("UTF-8")

        Returns
        -------
        `ExpProfile`
            The expression profile.
        """
        # "squeeze = True" ensures that a pd.read_tsv returns a series
        # as long as there is only one column
        e = cls(pd.read_csv(filepath_or_buffer, sep='\t',
                            index_col=0, header=0,
                            encoding=encoding, squeeze=True))

        if gene_table is not None:
            # filter genes
            e = e.filter_genes(gene_table.gene_names)

        return e