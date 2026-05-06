def crps(self):
        """
        Calculates the continuous ranked probability score.
        """
        return np.sum(self.errors["F_2"].values - self.errors["F_O"].values * 2.0 + self.errors["O_2"].values) / \
            (self.thresholds.size * self.num_forecasts)