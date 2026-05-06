def write_tsv(self, file_path: str, encoding: str = 'UTF-8',
                  sep: str = '\t'):
        """Write expression matrix to a tab-delimited text file.

        Parameters
        ----------
        file_path: str
            The path of the output file.
        encoding: str, optional
            The file encoding. ("UTF-8")

        Returns
        -------
        None
        """
        #if six.PY2:
        #    sep = sep.encode('UTF-8')

        self.to_csv(
            file_path, sep=sep, float_format='%.5f', mode='w',
            encoding=encoding, quoting=csv.QUOTE_NONE
        )

        _LOGGER.info('Wrote %d x %d expression matrix to "%s".',
                    self.p, self.n, file_path)