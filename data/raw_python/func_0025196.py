def calc_attribute_statistic(self, attribute, statistic, time):
        """
        Calculate statistics based on the values of an attribute. The following statistics are supported:
        mean, max, min, std, ptp (range), median, skew (mean - median), and percentile_(percentile value).

        Args:
            attribute: Attribute extracted from model grid
            statistic: Name of statistic being used.
            time: timestep of the object being investigated

        Returns:
            The value of the statistic
        """
        ti = np.where(self.times == time)[0][0]
        ma = np.where(self.masks[ti].ravel() == 1)
        if statistic in ['mean', 'max', 'min', 'std', 'ptp']:
            stat_val = getattr(self.attributes[attribute][ti].ravel()[ma], statistic)()
        elif statistic == 'median':
            stat_val = np.median(self.attributes[attribute][ti].ravel()[ma])
        elif statistic == "skew":
            stat_val = np.mean(self.attributes[attribute][ti].ravel()[ma]) - \
                       np.median(self.attributes[attribute][ti].ravel()[ma])
        elif 'percentile' in statistic:
            per = int(statistic.split("_")[1])
            stat_val = np.percentile(self.attributes[attribute][ti].ravel()[ma], per)
        elif 'dt' in statistic:
            stat_name = statistic[:-3]
            if ti == 0:
                stat_val = 0
            else:
                stat_val = self.calc_attribute_statistic(attribute, stat_name, time) \
                    - self.calc_attribute_statistic(attribute, stat_name, time - 1)
        else:
            stat_val = np.nan
        return stat_val