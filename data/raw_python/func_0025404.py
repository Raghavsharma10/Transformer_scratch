def crps_climo(self):
        """
        Calculate the climatological CRPS.
        """
        o_bar = self.errors["O"].values / float(self.num_forecasts)
        crps_c = np.sum(self.num_forecasts * (o_bar ** 2) - o_bar * self.errors["O"].values * 2.0 +
                        self.errors["O_2"].values) / float(self.thresholds.size * self.num_forecasts)
        return crps_c