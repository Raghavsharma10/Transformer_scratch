def fit(self, data):
        """Get the modality assignments of each splicing event in the data

        Parameters
        ----------
        data : pandas.DataFrame
            A (n_samples, n_events) dataframe of splicing events' PSI scores.
            Must be psi scores which range from 0 to 1

        Returns
        -------
        log2_bayes_factors : pandas.DataFrame
            A (n_modalities, n_events) dataframe of the estimated log2
            bayes factor for each splicing event, for each modality

        Raises
        ------
        AssertionError
            If any value in ``data`` does not fall only between 0 and 1.
        """
        self.assert_less_than_or_equal_1(data.values.flat)
        self.assert_non_negative(data.values.flat)

        if isinstance(data, pd.DataFrame):
            log2_bayes_factors = data.apply(self.single_feature_fit)
        elif isinstance(data, pd.Series):
            log2_bayes_factors = self.single_feature_fit(data)
        log2_bayes_factors.name = self.score_name
        return log2_bayes_factors