def predict(self, log2_bayes_factors, reset_index=False):
        """Guess the most likely modality for each event

        For each event that has at least one non-NA value, if no modalilites
        have logsumexp'd logliks greater than the log Bayes factor threshold,
        then they are assigned the 'multimodal' modality, because we cannot
        reject the null hypothesis that these did not come from the uniform
        distribution.

        Parameters
        ----------
        log2_bayes_factors : pandas.DataFrame
            A (4, n_events) dataframe with bayes factors for the Psi~1, Psi~0,
            bimodal, and middle modalities. If an event has no bayes factors
            for any of those modalities, it is ignored
        reset_index : bool
            If True, remove the first level of the index from the dataframe.
            Useful if you are using this function to apply to a grouped
            dataframe where the first level is something other than the
            modality, e.g. the celltype

        Returns
        -------
        modalities : pandas.Series
            A (n_events,) series with the most likely modality for each event

        """
        if reset_index:
            x = log2_bayes_factors.reset_index(level=0, drop=True)
        else:
            x = log2_bayes_factors
        if isinstance(x, pd.DataFrame):
            not_na = (x.notnull() > 0).any()
            not_na_columns = not_na[not_na].index
            x.ix[NULL_MODEL, not_na_columns] = self.logbf_thresh
        elif isinstance(x, pd.Series):
            x[NULL_MODEL] = self.logbf_thresh
        return x.idxmax()