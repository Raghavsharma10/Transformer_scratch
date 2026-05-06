def calc_timestep_statistic(self, statistic, time):
        """
        Calculate statistics from the primary attribute of the StObject.

        Args:
            statistic: statistic being calculated
            time: Timestep being investigated

        Returns:
            Value of the statistic
        """
        ti = np.where(self.times == time)[0][0]
        ma = np.where(self.masks[ti].ravel() == 1)
        if statistic in ['mean', 'max', 'min', 'std', 'ptp']:
            stat_val = getattr(self.timesteps[ti].ravel()[ma], statistic)()
        elif statistic == 'median':
            stat_val = np.median(self.timesteps[ti].ravel()[ma])
        elif 'percentile' in statistic:
            per = int(statistic.split("_")[1])
            stat_val = np.percentile(self.timesteps[ti].ravel()[ma], per)
        elif 'dt' in statistic:
            stat_name = statistic[:-3]
            if ti == 0:
                stat_val = 0
            else:
                stat_val = self.calc_timestep_statistic(stat_name, time) -\
                    self.calc_timestep_statistic(stat_name, time - 1)
        else:
            stat_val = np.nan
        return stat_val