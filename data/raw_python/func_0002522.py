def single_feature_logliks(self, feature):
        """Calculate log-likelihoods of each modality's parameterization

        Used for plotting the estimates of a single feature

        Parameters
        ----------
        featre : pandas.Series
            A single feature's values. All values must range from 0 to 1.

        Returns
        -------
        logliks : pandas.DataFrame
            The log-likelihood the data, for each model, for each
            parameterization

        Raises
        ------
        AssertionError
            If any value in ``x`` does not fall only between 0 and 1.
        """
        self.assert_less_than_or_equal_1(feature.values)
        self.assert_non_negative(feature.values)

        logliks = self._single_feature_logliks_one_step(
            feature, self.one_param_models)

        logsumexps = self.logliks_to_logsumexp(logliks)

        # If none of the one-parameter models passed, try the two-param models
        if (logsumexps <= self.logbf_thresh).all():
            logliks_two_params = self._single_feature_logliks_one_step(
                feature, self.two_param_models)
            logliks = pd.concat([logliks, logliks_two_params])
        return logliks