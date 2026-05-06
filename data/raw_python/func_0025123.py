def get_window_forecasts(self):
        """
        Aggregate the forecasts within the specified time windows.
        """
        for model_name in self.model_names:
            self.window_forecasts[model_name] = {}
            for size_threshold in self.size_thresholds:
                self.window_forecasts[model_name][size_threshold] = \
                    np.array([self.raw_forecasts[model_name][size_threshold][sl].sum(axis=0)
                              for sl in self.hour_windows])