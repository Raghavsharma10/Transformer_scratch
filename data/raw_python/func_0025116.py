def sample_forecast_max_hail(self, dist_model_name, condition_model_name,
                                 num_samples, condition_threshold=0.5, query=None):
        """
        Samples every forecast hail object and returns an empirical distribution of possible maximum hail sizes.

        Hail sizes are sampled from each predicted gamma distribution. The total number of samples equals
        num_samples * area of the hail object. To get the maximum hail size for each realization, the maximum
        value within each area sample is used.

        Args:
            dist_model_name: Name of the distribution machine learning model being evaluated
            condition_model_name: Name of the hail/no-hail model being evaluated
            num_samples: Number of maximum hail samples to draw
            condition_threshold: Threshold for drawing hail samples
            query: A str that selects a subset of the data for evaluation

        Returns:
            A numpy array containing maximum hail samples for each forecast object.
        """
        if query is not None:
            dist_forecasts = self.matched_forecasts["dist"][dist_model_name].query(query)
            dist_forecasts = dist_forecasts.reset_index(drop=True)
            condition_forecasts = self.matched_forecasts["condition"][condition_model_name].query(query)
            condition_forecasts = condition_forecasts.reset_index(drop=True)
        else:
            dist_forecasts = self.matched_forecasts["dist"][dist_model_name]
            condition_forecasts = self.matched_forecasts["condition"][condition_model_name]
        max_hail_samples = np.zeros((dist_forecasts.shape[0], num_samples))
        areas = dist_forecasts["Area"].values
        for f in np.arange(dist_forecasts.shape[0]):
            condition_prob = condition_forecasts.loc[f, self.forecast_bins["condition"][0]]
            if condition_prob >= condition_threshold:
                max_hail_samples[f] = np.sort(gamma.rvs(*dist_forecasts.loc[f, self.forecast_bins["dist"]].values,
                                                        size=(num_samples, areas[f])).max(axis=1))
        return max_hail_samples