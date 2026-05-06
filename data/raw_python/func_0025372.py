def point_probability(self, threshold):
        """
        Determine the probability of exceeding a threshold at a grid point based on the ensemble forecasts at
        that point.

        Args:
            threshold: If >= threshold assigns a 1 to member, otherwise 0.

        Returns:
            EnsembleConsensus
        """
        point_prob = np.zeros(self.data.shape[1:])
        for t in range(self.data.shape[1]):
            point_prob[t] = np.where(self.data[:, t] >= threshold, 1.0, 0.0).mean(axis=0)
        return EnsembleConsensus(point_prob, "point_probability", self.ensemble_name,
                                 self.run_date, self.variable + "_{0:0.2f}_{1}".format(threshold,
                                                                                       self.units.replace(" ", "_")),
                                 self.start_date, self.end_date, "")