def predictions(self, setup, n_jobs=-1):
        """
        Returns a `pandas.DataFrame`_ with the weighted average predictions and variance of all readouts for each possible
        clampings in the given experimental setup.
        For each logical network the weight corresponds to the number of networks having the same behavior.

        Parameters
        ----------
        setup : :class:`caspo.core.setup.Setup`
            Experimental setup

        n_jobs : int
            Number of jobs to run in parallel. Default to -1 (all cores available)

        Returns
        -------
        `pandas.DataFrame`_
            DataFrame with the weighted average predictions and variance of all readouts for each possible clamping


        .. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe

        .. seealso:: `Wikipedia: Weighted sample variance <https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance>`_
        """
        stimuli, inhibitors, readouts = setup.stimuli, setup.inhibitors, setup.readouts
        nc = len(setup.cues())
        predictions = np.zeros((len(self), 2**nc, len(setup)))
        predictions[:, :, :] = Parallel(n_jobs=n_jobs)(delayed(__parallel_predictions__)(n, list(setup.clampings_iter(setup.cues())), readouts, stimuli, inhibitors) for n in self)

        avg = np.average(predictions[:, :, nc:], axis=0, weights=self.__networks)
        var = np.average((predictions[:, :, nc:]-avg)**2, axis=0, weights=self.__networks)

        rcues = ["TR:%s" % c for c in setup.cues(True)]
        cols = np.concatenate([rcues, ["AVG:%s" % r for r in readouts], ["VAR:%s" % r for r in readouts]])

        #use the first network predictions to extract all clampings
        df = pd.DataFrame(np.concatenate([predictions[0, :, :nc], avg, var], axis=1), columns=cols)
        df[rcues] = df[rcues].astype(int)

        return df