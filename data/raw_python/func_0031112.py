def read_tsv(cls, path, encoding='utf-8'):
        """Read a gene set database from a tab-delimited text file.

        Parameters
        ----------
        path: str
            The path name of the the file.
        encoding: str
            The encoding of the text file.

        Returns
        -------
        None
        """
        gene_sets = []
        n = 0
        with open(path, 'rb') as fh:
            reader = csv.reader(fh, dialect='excel-tab', encoding=encoding)
            for l in reader:
                n += 1
                gs = GeneSet.from_list(l)
                gene_sets.append(gs)
        logger.debug('Read %d gene sets.', n)
        logger.debug('Size of gene set list: %d', len(gene_sets))
        return cls(gene_sets)