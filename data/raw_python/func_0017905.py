def shallow_run(self):
        """Derive not-period-based features."""
        # Number of data points
        self.n_points = len(self.date)

        # Weight calculation.
        # All zero values.
        if not self.err.any():
            self.err = np.ones(len(self.mag)) * np.std(self.mag)
        # Some zero values.
        elif not self.err.all():
            np.putmask(self.err, self.err==0, np.median(self.err))

        self.weight = 1. / self.err
        self.weighted_sum = np.sum(self.weight)

        # Simple statistics, mean, median and std.
        self.mean = np.mean(self.mag)
        self.median = np.median(self.mag)
        self.std = np.std(self.mag)

        # Weighted mean and std.
        self.weighted_mean = np.sum(self.mag * self.weight) / self.weighted_sum
        self.weighted_std = np.sqrt(np.sum((self.mag - self.weighted_mean) ** 2 \
                                           * self.weight) / self.weighted_sum)

        # Skewness and kurtosis.
        self.skewness = ss.skew(self.mag)
        self.kurtosis = ss.kurtosis(self.mag)

        # Normalization-test. Shapiro-Wilk test.
        shapiro = ss.shapiro(self.mag)
        self.shapiro_w = shapiro[0]
        # self.shapiro_log10p = np.log10(shapiro[1])

        # Percentile features.
        self.quartile31 = np.percentile(self.mag, 75) \
                          - np.percentile(self.mag, 25)

        # Stetson K.
        self.stetson_k = self.get_stetson_k(self.mag, self.median, self.err)

        # Ratio between higher and lower amplitude than average.
        self.hl_amp_ratio = self.half_mag_amplitude_ratio(
            self.mag, self.median, self.weight)
        # This second function's value is very similar with the above one.
        # self.hl_amp_ratio2 = self.half_mag_amplitude_ratio2(
        #    self.mag, self.median)

        # Cusum
        self.cusum = self.get_cusum(self.mag)

        # Eta
        self.eta = self.get_eta(self.mag, self.weighted_std)