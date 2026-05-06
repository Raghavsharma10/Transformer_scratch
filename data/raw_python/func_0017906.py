def deep_run(self):
        """Derive period-based features."""
        # Lomb-Scargle period finding.
        self.get_period_LS(self.date, self.mag, self.n_threads, self.min_period)

        # Features based on a phase-folded light curve
        # such as Eta, slope-percentile, etc.
        # Should be called after the getPeriodLS() is called.

        # Created phased a folded light curve.
        # We use period * 2 to take eclipsing binaries into account.
        phase_folded_date = self.date % (self.period * 2.)
        sorted_index = np.argsort(phase_folded_date)

        folded_date = phase_folded_date[sorted_index]
        folded_mag = self.mag[sorted_index]

        # phase Eta
        self.phase_eta = self.get_eta(folded_mag, self.weighted_std)

        # Slope percentile.
        self.slope_per10, self.slope_per90 = \
            self.slope_percentile(folded_date, folded_mag)

        # phase Cusum
        self.phase_cusum = self.get_cusum(folded_mag)