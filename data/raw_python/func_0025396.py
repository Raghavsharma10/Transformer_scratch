def update(self, forecasts, observations):
        """
        Update the statistics with a set of forecasts and observations.

        Args:
            forecasts (numpy.ndarray): Array of forecast probability values
            observations (numpy.ndarray): Array of observation values
        """
        for t, threshold in enumerate(self.thresholds[:-1]):
            self.frequencies.loc[t, "Positive_Freq"] += np.count_nonzero((threshold <= forecasts) &
                                                                         (forecasts < self.thresholds[t+1]) &
                                                                         (observations >= self.obs_threshold))
            self.frequencies.loc[t, "Total_Freq"] += np.count_nonzero((threshold <= forecasts) &
                                                                      (forecasts < self.thresholds[t+1]))