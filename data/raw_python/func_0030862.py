def read_tsv(cls, file_path: str, gene_table: ExpGeneTable = None,
                 encoding: str = 'UTF-8', sep: str = '\t'):
        """Read expression matrix from a tab-delimited text file.

        Parameters
        ----------
        file_path: str
            The path of the text file.
        gene_table: `ExpGeneTable` object, optional
            The set of valid genes. If given, the genes in the text file will
            be filtered against this set of genes. (None)
        encoding: str, optional
            The file encoding. ("UTF-8")
        sep: str, optional
            The separator. ("\t")

        Returns
        -------
        `ExpMatrix`
            The expression matrix.
        """
        # use pd.read_csv to parse the tsv file into a DataFrame
        matrix = cls(pd.read_csv(file_path, sep=sep, index_col=0, header=0,
                                 encoding=encoding))

        # parse index column separately
        # (this seems to be the only way we can prevent pandas from converting
        #  "nan" or "NaN" to floats in the index)['1_cell_306.120', '1_cell_086.024', '1_cell_168.103']
        #ind = pd.read_csv(file_path, sep=sep, usecols=[0, ], header=0,
        #                  encoding=encoding, na_filter=False)
        ind = pd.read_csv(file_path, sep=sep, usecols=[0, ], header=None,
                          skiprows=1, encoding=encoding, na_filter=False)

        matrix.index = ind.iloc[:, 0]
        matrix.index.name = 'Genes'

        if gene_table is not None:
            # filter genes
            matrix = matrix.filter_genes(gene_table.gene_names)

        return matrix