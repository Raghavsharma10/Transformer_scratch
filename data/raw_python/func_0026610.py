def set_epoch(self, year):
        """Updates the epoch for all subsequent conversions.

        Parameters
        ==========
        year : float
            Decimal year

        """

        fa.loadapxsh(self.datafile, np.float(year))
        self.year = year