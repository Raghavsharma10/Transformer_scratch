def update(self, forecasts, observations):
        """
        Update the ROC curve with a set of forecasts and observations

        Args:
            forecasts: 1D array of forecast values
            observations: 1D array of observation values.
        """
        for t, threshold in enumerate(self.thresholds):
            tp = np.count_nonzero((forecasts >= threshold) & (observations >= self.obs_threshold))
            fp = np.count_nonzero((forecasts >= threshold) &
                                  (observations < self.obs_threshold))
            fn = np.count_nonzero((forecasts < threshold) &
                                  (observations >= self.obs_threshold))
            tn = np.count_nonzero((forecasts < threshold) &
                                  (observations < self.obs_threshold))
            self.contingency_tables.iloc[t] += [tp, fp, fn, tn]