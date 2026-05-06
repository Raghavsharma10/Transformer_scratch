def load_forecasts(self):
        """
        Load neighborhood probability forecasts.
        """
        run_date_str = self.run_date.strftime("%Y%m%d")
        forecast_file = self.forecast_path + "{0}/{1}_{2}_{3}_consensus_{0}.nc".format(run_date_str,
                                                                                       self.ensemble_name,
                                                                                       self.model_name,
                                                                                       self.forecast_variable)
        print("Forecast file: " + forecast_file)
        forecast_data = Dataset(forecast_file)
        for size_threshold in self.size_thresholds:
            for smoothing_radius in self.smoothing_radii:
                for neighbor_radius in self.neighbor_radii:
                    hour_var = "neighbor_prob_r_{0:d}_s_{1:d}_{2}_{3:0.2f}".format(neighbor_radius, smoothing_radius,
                                                                                   self.forecast_variable,
                                                                                   float(size_threshold))
                    period_var = "neighbor_prob_{0:d}-hour_r_{1:d}_s_{2:d}_{3}_{4:0.2f}".format(self.end_hour -
                                                                                                self.start_hour + 1,
                                                                                                neighbor_radius,
                                                                                                smoothing_radius,
                                                                                                self.forecast_variable,
                                                                                                float(size_threshold))

                    print("Loading forecasts {0} {1} {2} {3} {4}".format(self.run_date, self.model_name,
                                                                         self.forecast_variable, size_threshold,
                                                                         smoothing_radius))
                    if hour_var in forecast_data.variables.keys():
                        self.hourly_forecasts[hour_var] = forecast_data.variables[hour_var][:]
                    if period_var in forecast_data.variables.keys():
                        self.period_forecasts[period_var] = forecast_data.variables[period_var][:]
        forecast_data.close()