def load_data(self, grid_method="gamma", num_samples=1000, condition_threshold=0.5, zero_inflate=False,
                  percentile=None):
        """
        Reads the track forecasts and converts them to grid point values based on random sampling.

        Args:
            grid_method: "gamma" by default
            num_samples: Number of samples drawn from predicted pdf
            condition_threshold: Objects are not written to the grid if condition model probability is below this
                threshold.
            zero_inflate: Whether to sample zeros from a Bernoulli sampler based on the condition model probability
            percentile: If None, outputs the mean of the samples at each grid point, otherwise outputs the specified
                percentile from 0 to 100.

        Returns:
            0 if tracks are successfully sampled on to grid. If no tracks are found, returns -1.
        """
        self.percentile = percentile
        if self.track_forecasts == {}:
            self.load_track_forecasts()
        if self.track_forecasts == {}:
            return -1
        if self.data is None:
            self.data = np.zeros((len(self.members), self.times.size, self.grid_shape[0], self.grid_shape[1]),
                                 dtype=np.float32)
        else:
            self.data[:] = 0
        if grid_method in ["mean", "median", "samples"]:
            for m, member in enumerate(self.members):
                print("Sampling " + member)
                for track_forecast in self.track_forecasts[member]:
                    times = track_forecast["properties"]["times"]
                    for s, step in enumerate(track_forecast["features"]):
                        forecast_pdf = np.array(step['properties'][self.variable + "_" +
                                                                   self.ensemble_name.replace(" ", "-")])
                        forecast_time = self.run_date + timedelta(hours=times[s])
                        t = np.where(self.times == forecast_time)[0][0]
                        mask = np.array(step['properties']["masks"], dtype=int)
                        i = np.array(step['properties']["i"], dtype=int)
                        i = i[mask == 1]
                        j = np.array(step['properties']["j"], dtype=int)
                        j = j[mask == 1]
                        if grid_method == "samples":
                            intensities = np.array(step["properties"]["timesteps"], dtype=float)[mask == 1]
                            rankings = np.argsort(intensities)
                            samples = np.random.choice(self.forecast_bins, size=intensities.size, replace=True,
                                                       p=forecast_pdf)
                            self.data[m, t, i[rankings], j[rankings]] = samples
                        else:
                            if grid_method == "mean":
                                forecast_value = np.sum(forecast_pdf * self.forecast_bins)
                            elif grid_method == "median":
                                forecast_cdf = np.cumsum(forecast_pdf)
                                forecast_value = self.forecast_bins[np.argmin(np.abs(forecast_cdf - 0.5))]
                            else:
                                forecast_value = 0
                            self.data[m, t, i, j] = forecast_value
        if grid_method in ["gamma"]:
            full_condition_name = "condition_" + self.condition_model_name.replace(" ", "-")
            dist_model_name = self.variable + "_" + self.ensemble_name.replace(" ", "-")
            for m, member in enumerate(self.members):
                for track_forecast in self.track_forecasts[member]:
                    times = track_forecast["properties"]["times"]
                    for s, step in enumerate(track_forecast["features"]):
                        forecast_params = step["properties"][dist_model_name]
                        if self.condition_model_name is not None:
                            condition = step["properties"][full_condition_name]
                        else:
                            condition = None
                        forecast_time = self.run_date + timedelta(hours=times[s])
                        if forecast_time in self.times:
                            t = np.where(self.times == forecast_time)[0][0]
                            mask = np.array(step["properties"]["masks"], dtype=int)
                            rankings = np.argsort(step["properties"]["timesteps"])[mask == 1]
                            i = np.array(step["properties"]["i"], dtype=int)[mask == 1][rankings]
                            j = np.array(step["properties"]["j"], dtype=int)[mask == 1][rankings]
                            if rankings.size > 0:
                                raw_samples = np.sort(gamma.rvs(forecast_params[0], loc=forecast_params[1],
                                                                scale=forecast_params[2],
                                                                size=(num_samples, rankings.size)),
                                                      axis=1)
                                if zero_inflate:
                                    raw_samples *= bernoulli.rvs(condition,
                                                                 size=(num_samples, rankings.size))
                                if percentile is None:
                                    samples = raw_samples.mean(axis=0)
                                else:
                                    samples = np.percentile(raw_samples, percentile, axis=0)
                                if condition is None or condition >= condition_threshold:
                                    self.data[m, t, i, j] = samples
        return 0