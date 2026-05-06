def crps(self, model_type, model_name, condition_model_name, condition_threshold, query=None):
        """
        Calculates the cumulative ranked probability score (CRPS) on the forecast data.

        Args:
            model_type: model type being evaluated.
            model_name: machine learning model being evaluated.
            condition_model_name: Name of the hail/no-hail model being evaluated
            condition_threshold: Threshold for using hail size CDF
            query: pandas query string to filter the forecasts based on the metadata


        Returns:
            a DistributedCRPS object
        """

        def gamma_cdf(x, a, loc, b):
            if a == 0 or b == 0:
                cdf = np.ones(x.shape)
            else:
                cdf = gamma.cdf(x, a, loc, b)
            return cdf

        crps_obj = DistributedCRPS(self.dist_thresholds)
        if query is not None:
            sub_forecasts = self.matched_forecasts[model_type][model_name].query(query)
            sub_forecasts = sub_forecasts.reset_index(drop=True)
            condition_forecasts = self.matched_forecasts["condition"][condition_model_name].query(query)
            condition_forecasts = condition_forecasts.reset_index(drop=True)
        else:
            sub_forecasts = self.matched_forecasts[model_type][model_name]
            condition_forecasts = self.matched_forecasts["condition"][condition_model_name]
        if sub_forecasts.shape[0] > 0:
            if model_type == "dist":
                forecast_cdfs = np.zeros((sub_forecasts.shape[0], self.dist_thresholds.size))
                for f in range(sub_forecasts.shape[0]):
                    condition_prob = condition_forecasts.loc[f, self.forecast_bins["condition"][0]]
                    if condition_prob >= condition_threshold:
                        f_params = [0, 0, 0]
                    else:
                        f_params = sub_forecasts[self.forecast_bins[model_type]].values[f]
                    forecast_cdfs[f] = gamma_cdf(self.dist_thresholds, f_params[0], f_params[1], f_params[2])
                obs_cdfs = np.array([gamma_cdf(self.dist_thresholds, *params)
                                    for params in sub_forecasts[self.type_cols[model_type]].values])
                crps_obj.update(forecast_cdfs, obs_cdfs)
            else:
                crps_obj.update(sub_forecasts[self.forecast_bins[model_type].astype(str)].values,
                                sub_forecasts[self.type_cols[model_type]].values)

        return crps_obj