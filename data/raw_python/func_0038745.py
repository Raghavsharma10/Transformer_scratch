def to_dataframe(self, networks=False, dataset=None, size=False, n_jobs=-1):
        """
        Converts the list of logical networks to a `pandas.DataFrame`_ object instance

        Parameters
        ----------
        networks : boolean
            If True, a column with number of networks having the same behavior is included in the DataFrame

        dataset: Optional[:class:`caspo.core.dataset.Dataset`]
            If not None, a column with the MSE with respect to the given dataset is included in the DataFrame

        size: boolean
            If True, a column with the size of each logical network is included in the DataFrame

        n_jobs : int
            Number of jobs to run in parallel. Default to -1 (all cores available)

        Returns
        -------
        `pandas.DataFrame`_
            DataFrame representation of the list of logical networks.


        .. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
        """
        length = len(self)
        df = pd.DataFrame(self.__matrix, columns=map(str, self.hg.mappings))

        if networks:
            df = pd.concat([df, pd.DataFrame({'networks': self.__networks})], axis=1)

        if dataset is not None:
            clampings = dataset.clampings
            readouts = dataset.readouts.columns
            observations = dataset.readouts.values
            pos = ~np.isnan(observations)

            mse = Parallel(n_jobs=n_jobs)(delayed(__parallel_mse__)(n, clampings, readouts, observations[pos], pos) for n in self)
            df = pd.concat([df, pd.DataFrame({'mse': mse})], axis=1)

        if size:
            df = pd.concat([df, pd.DataFrame({'size': np.fromiter((n.size for n in self), int, length)})], axis=1)

        return df