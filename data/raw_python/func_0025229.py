def evaluate_period_forecasts(self):
        """
        Evaluates ROC and Reliability scores for forecasts over the full period from start hour to end hour

        Returns:
            A pandas DataFrame with full-period metadata and verification statistics
        """
        score_columns = ["Run_Date", "Ensemble Name", "Model_Name", "Forecast_Variable", "Neighbor_Radius",
                         "Smoothing_Radius", "Size_Threshold",  "ROC", "Reliability"]
        all_scores = pd.DataFrame(columns=score_columns)
        if self.coordinate_file is not None:
            coord_mask = np.where((self.coordinates["lon"] >= self.lon_bounds[0]) &
                                  (self.coordinates["lon"] <= self.lon_bounds[1]) &
                                  (self.coordinates["lat"] >= self.lat_bounds[0]) &
                                  (self.coordinates["lat"] <= self.lat_bounds[1]) &
                                  (self.period_obs[self.mask_variable] > 0))
        else:
            coord_mask = None
        for neighbor_radius in self.neighbor_radii:
            n_filter = disk(neighbor_radius)
            for s, size_threshold in enumerate(self.size_thresholds):
                period_obs = fftconvolve(self.period_obs[self.mrms_variable] >= self.obs_thresholds[s],
                                         n_filter, mode="same")
                period_obs[period_obs > 1] = 1
                if self.obs_mask and self.coordinate_file is None:
                    period_obs = period_obs[self.period_obs[self.mask_variable] > 0]
                elif self.obs_mask and self.coordinate_file is not None:
                    period_obs = period_obs[coord_mask[0], coord_mask[1]]
                else:
                    period_obs = period_obs.ravel()
                for smoothing_radius in self.smoothing_radii:
                    print("Eval period forecast {0} {1} {2} {3} {4} {5}".format(self.model_name,
                                                                                self.forecast_variable,
                                                                                self.run_date,
                                                                                neighbor_radius,
                                                                                size_threshold, smoothing_radius))
                    period_var = "neighbor_prob_{0:d}-hour_r_{1:d}_s_{2:d}_{3}_{4:0.2f}".format(self.end_hour -
                                                                                                self.start_hour + 1,
                                                                                                neighbor_radius,
                                                                                                smoothing_radius,
                                                                                                self.forecast_variable,
                                                                                                size_threshold)
                    if self.obs_mask and self.coordinate_file is None:
                        period_forecast = self.period_forecasts[period_var][self.period_obs[self.mask_variable] > 0]
                    elif self.obs_mask and self.coordinate_file is not None:
                        period_forecast = self.period_forecasts[period_var][coord_mask[0], coord_mask[1]]
                    else:
                        period_forecast = self.period_forecasts[period_var].ravel()
                    roc = DistributedROC(thresholds=self.probability_levels, obs_threshold=0.5)
                    roc.update(period_forecast, period_obs)
                    rel = DistributedReliability(thresholds=self.probability_levels, obs_threshold=0.5)
                    rel.update(period_forecast, period_obs)
                    row = [self.run_date, self.ensemble_name, self.model_name, self.forecast_variable, neighbor_radius,
                           smoothing_radius, size_threshold, roc, rel]
                    all_scores.loc[period_var] = row
        return all_scores