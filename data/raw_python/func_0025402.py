def update(self, forecasts, observations):
        """
        Update the statistics with forecasts and observations.

        Args:
            forecasts: The discrete Cumulative Distribution Functions of
            observations:
        """
        if len(observations.shape) == 1:
            obs_cdfs = np.zeros((observations.size, self.thresholds.size))
            for o, observation in enumerate(observations):
                obs_cdfs[o, self.thresholds >= observation] = 1
        else:
            obs_cdfs = observations
        self.errors["F_2"] += np.sum(forecasts ** 2, axis=0)
        self.errors["F_O"] += np.sum(forecasts * obs_cdfs, axis=0)
        self.errors["O_2"] += np.sum(obs_cdfs ** 2, axis=0)
        self.errors["O"] += np.sum(obs_cdfs, axis=0)
        self.num_forecasts += forecasts.shape[0]