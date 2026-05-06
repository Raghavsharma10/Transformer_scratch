def read_pickle(fn):
        """Load a GOParser object from a pickle file.

        The function automatically detects whether the file is compressed
        with gzip.

        Parameters
        ----------
        fn: str
            Path of the pickle file.

        Returns
        -------
        `GOParser`
            The GOParser object stored in the pickle file.
        """
        with misc.open_plain_or_gzip(fn, 'rb') as fh:
            parser = pickle.load(fh)
        return parser