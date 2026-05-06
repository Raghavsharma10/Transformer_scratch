def differences(self, networks, readouts, prepend=""):
        """
        Returns the total number of pairwise differences over the given readouts for the given networks

        Parameters
        ----------
        networks : iterable[:class:`caspo.core.logicalnetwork.LogicalNetwork`]
            Iterable of logical networks to compute pairwise differences

        readouts : list[str]
            List of readouts species names

        prepend : str
            Columns are renamed using the given string at the beginning


        Returns
        -------
        `pandas.DataFrame`_
            Total number of pairwise differences for each clamping over each readout


        .. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
        """
        z, p = np.zeros((len(self), len(readouts)), dtype=int), np.zeros(len(self), dtype=int)
        for n1, n2 in it.combinations(networks, 2):
            r, c = np.where(n1.predictions(self, readouts) != n2.predictions(self, readouts))
            z[r, c] += 1
            p[r] += 1

        df = pd.DataFrame(z, columns=[prepend + "%s" % c for c in readouts])
        return pd.concat([df, pd.Series(p, name='pairs')], axis=1)