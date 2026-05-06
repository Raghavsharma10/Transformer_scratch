def write_tsv(self, path, encoding='UTF-8'):
        """Write expression matrix to a tab-delimited text file.

        Parameters
        ----------
        path: str
            The path of the output file.
        encoding: str, optional
            The file encoding. ("UTF-8")

        Returns
        -------
        None
        """
        assert isinstance(path, (str, _oldstr))
        assert isinstance(encoding, (str, _oldstr))

        sep = '\t'
        if six.PY2:
            sep = sep.encode('UTF-8')

        self.to_csv(
            path, sep=sep, float_format='%.5f', mode='w',
            encoding=encoding, header=True
        )

        logger.info('Wrote expression profile "%s" with %d genes to "%s".',
                    self.name, self.p, path)