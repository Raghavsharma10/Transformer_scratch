def evaluate_hourly_forecasts(self):
        """
        Calculates ROC curves and Reliability scores for each forecast hour.

        Returns:
            A pandas DataFrame containing forecast metadata as well as DistributedROC and Reliability objects.
        """
        score_columns = ["Run_Date", "Forecast_Hour", "Ensemble Name", "Model_Name", "Forecast_Variable",
                         "Neighbor_Radius", "Smoothing_Radius", "Size_Threshold", "ROC", "Reliability"]
        all_scores = pd.DataFrame(columns=score_columns)
        for h, hour in enumerate(range(self.start_hour, self.end_hour + 1)):
            for neighbor_radius in self.neighbor_radii:
                n_filter = disk(neighbor_radius)
                for s, size_threshold in enumerate(self.size_thresholds):
                    print("Eval hourly forecast {0:02d} {1} {2} {3} {4:d} {5:d}".format(hour, self.model_name,
                                                                                        self.forecast_variable,
                                                                                        self.run_date, neighbor_radius,
                                                                                        size_threshold))
                    hour_obs = fftconvolve(self.raw_obs[self.mrms_variable][h] >= self.obs_thresholds[s],
                                           n_filter, mode="same")
                    hour_obs[hour_obs > 1] = 1
                    hour_obs[hour_obs < 1] = 0
                    if self.obs_mask:
                        hour_obs = hour_obs[self.raw_obs[self.mask_variable][h] > 0]
                    for smoothing_radius in self.smoothing_radii:
                        hour_var = "neighbor_prob_r_{0:d}_s_{1:d}_{2}_{3:0.2f}".format(neighbor_radius,
                                                                                       smoothing_radius,
                                                                                       self.forecast_variable,
                                                                                       size_threshold)
                        if self.obs_mask:
                            hour_forecast = self.hourly_forecasts[hour_var][h][self.raw_obs[self.mask_variable][h] > 0]
                        else:
                            hour_forecast = self.hourly_forecasts[hour_var][h]
                        roc = DistributedROC(thresholds=self.probability_levels, obs_threshold=0.5)
                        roc.update(hour_forecast, hour_obs)
                        rel = DistributedReliability(thresholds=self.probability_levels, obs_threshold=0.5)
                        rel.update(hour_forecast, hour_obs)
                        row = [self.run_date, hour, self.ensemble_name, self.model_name, self.forecast_variable,
                               neighbor_radius,
                               smoothing_radius, size_threshold, roc, rel]
                        all_scores.loc[hour_var + "_{0:d}".format(hour)] = row
        return all_scores