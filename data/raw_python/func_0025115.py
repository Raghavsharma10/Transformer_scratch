def roc(self, model_type, model_name, intensity_threshold, prob_thresholds, query=None):
        """
        Calculates a ROC curve at a specified intensity threshold.

        Args:
            model_type: type of model being evaluated (e.g. size).
            model_name: machine learning model being evaluated
            intensity_threshold: forecast bin used as the split point for evaluation
            prob_thresholds: Array of probability thresholds being evaluated.
            query: str to filter forecasts based on values of forecasts, obs, and metadata.

        Returns:
             A DistributedROC object
        """
        roc_obj = DistributedROC(prob_thresholds, 0.5)
        if query is not None:
            sub_forecasts = self.matched_forecasts[model_type][model_name].query(query)
            sub_forecasts = sub_forecasts.reset_index(drop=True)
        else:
            sub_forecasts = self.matched_forecasts[model_type][model_name]
        obs_values = np.zeros(sub_forecasts.shape[0])
        if sub_forecasts.shape[0] > 0:
            if model_type == "dist":
                forecast_values = np.array([gamma_sf(intensity_threshold, *params)
                                            for params in sub_forecasts[self.forecast_bins[model_type]].values])
                obs_probs = np.array([gamma_sf(intensity_threshold, *params)
                                    for params in sub_forecasts[self.type_cols[model_type]].values])
                obs_values[obs_probs >= 0.01] = 1
            elif len(self.forecast_bins[model_type]) > 1:
                fbin = np.argmin(np.abs(self.forecast_bins[model_type] - intensity_threshold))
                forecast_values = 1 - sub_forecasts[self.forecast_bins[model_type].astype(str)].values.cumsum(axis=1)[:,
                                    fbin]
                obs_values[sub_forecasts[self.type_cols[model_type]].values >= intensity_threshold] = 1
            else:
                forecast_values = sub_forecasts[self.forecast_bins[model_type].astype(str)[0]].values
                obs_values[sub_forecasts[self.type_cols[model_type]].values >= intensity_threshold] = 1
            roc_obj.update(forecast_values, obs_values)
        return roc_obj