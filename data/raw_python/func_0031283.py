def write_pickle(self, path, compress=False):
        """Serialize the current `GOParser` object and store it in a pickle file.

        Parameters
        ----------
        path: str
            Path of the output file.
        compress: bool, optional
            Whether to compress the file using gzip.

        Returns
        -------
        None

        Notes
        -----
        Compression with gzip is significantly slower than storing the file
        in uncompressed form.
        """
        logger.info('Writing pickle to "%s"...', path)
        if compress:
            with gzip.open(path, 'wb') as ofh:
                pickle.dump(self, ofh, pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, 'wb') as ofh:
                pickle.dump(self, ofh, pickle.HIGHEST_PROTOCOL)