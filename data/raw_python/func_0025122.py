def load_forecasts(self):
        """
        Load the forecast files into memory.
        """
        run_date_str = self.run_date.strftime("%Y%m%d")
        for model_name in self.model_names:
            self.raw_forecasts[model_name] = {}
            forecast_file = self.forecast_path + run_date_str + "/" + \
                model_name.replace(" ", "-") + "_hailprobs_{0}_{1}.nc".format(self.ensemble_member, run_date_str)
            forecast_obj = Dataset(forecast_file)
            forecast_hours = forecast_obj.variables["forecast_hour"][:]
            valid_hour_indices = np.where((self.start_hour <= forecast_hours) & (forecast_hours <= self.end_hour))[0]
            for size_threshold in self.size_thresholds:
                self.raw_forecasts[model_name][size_threshold] = \
                    forecast_obj.variables["prob_hail_{0:02d}_mm".format(size_threshold)][valid_hour_indices]
            forecast_obj.close()