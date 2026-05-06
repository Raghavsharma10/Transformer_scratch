def load_data(self, num_samples=1000, percentiles=None):
        """
        Args:
            num_samples: Number of random samples at each grid point
            percentiles: Which percentiles to extract from the random samples

        Returns:
        """
        self.percentiles = percentiles
        self.num_samples = num_samples
        if self.model_name.lower() in ["wrf"]:
            mo = ModelOutput(self.ensemble_name, self.member, self.run_date, self.variable,
                             self.start_date, self.end_date, self.path, self.map_file, self.single_step)
            mo.load_data()
            self.data = mo.data[:]
            if mo.units == "m":
                self.data *= 1000
                self.units = "mm"
            else:
                self.units = mo.units
        else:
            if self.track_forecasts is None:
                self.load_track_data()
            self.units = "mm"
            self.data = np.zeros((self.forecast_hours.size,
                                  self.mapping_data["lon"].shape[0],
                                  self.mapping_data["lon"].shape[1]), dtype=np.float32)
            
            if self.percentiles is not None:
                self.percentile_data = np.zeros([len(self.percentiles)] + list(self.data.shape))
            full_condition_name = "condition_" + self.condition_model_name.replace(" ", "-")
            dist_model_name = "dist" + "_" + self.model_name.replace(" ", "-")
            for track_forecast in self.track_forecasts:
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
                        mask = np.array(step["properties"]["masks"], dtype=int).ravel()
                        rankings = np.argsort(np.array(step["properties"]["timesteps"]).ravel()[mask==1])
                        i = np.array(step["properties"]["i"], dtype=int).ravel()[mask == 1][rankings]
                        j = np.array(step["properties"]["j"], dtype=int).ravel()[mask == 1][rankings]
                        if rankings.size > 0 and forecast_params[0] > 0.1 and 1 < forecast_params[2] < 100:
                            raw_samples = np.sort(gamma.rvs(forecast_params[0], loc=forecast_params[1],
                                                            scale=forecast_params[2],
                                                            size=(num_samples, rankings.size)),
                                                  axis=1)
                            if self.percentiles is None:
                                samples = raw_samples.mean(axis=0)
                                if condition >= self.condition_threshold:
                                    self.data[t, i, j] = samples
                            else:
                                for p, percentile in enumerate(self.percentiles):
                                    if percentile != "mean":
                                        if condition >= self.condition_threshold:
                                            self.percentile_data[p, t, i, j] = np.percentile(raw_samples, percentile,
                                                                                             axis=0)
                                    else:
                                        if condition >= self.condition_threshold:
                                            self.percentile_data[p, t, i, j] = np.mean(raw_samples, axis=0)
                                samples = raw_samples.mean(axis=0)
                                if condition >= self.condition_threshold:
                                    self.data[t, i, j] = samples