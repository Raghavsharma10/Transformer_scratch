def get_pretty_format(self, max_name_length=0):
        """Returns a nicely formatted string describing the result.

        Parameters
        ----------
        max_name_length: int [0]
            The maximum length of the gene set name (in characters). If the
            gene set name is longer than this number, it will be truncated and
            "..." will be appended to it, so that the final string exactly
            meets the length requirement. If 0 (default), no truncation is
            performed. If not 0, must be at least 3.

        Returns
        -------
        str
            The formatted string.

        Raises
        ------
        ValueError
            If an invalid length value is specified.
        """

        assert isinstance(max_name_length, (int, np.integer))

        if max_name_length < 0 or (1 <= max_name_length <= 2):
            raise ValueError('max_name_length must be 0 or >= 3.')

        gs_name = self.gene_set._name
        if max_name_length > 0 and len(gs_name) > max_name_length:
            assert max_name_length >= 3
            gs_name = gs_name[:(max_name_length - 3)] + '...'

        param_str = '(%d/%d @ %d/%d, pval=%.1e, fe=%.1fx)' \
                % (self.k, self.K, self.n, self.N,
                   self.pval, self.fold_enrichment)

        return '%s %s' % (gs_name, param_str)