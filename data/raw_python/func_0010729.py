def calc_precipitation_stats(self, months=None, avg_stats=True, percentile=50):
        """
        Calculates precipitation statistics for the cascade model while aggregating hourly observations

        Parameters
        ----------
        months :        Months for each seasons to be used for statistics (array of numpy array, default=1-12, e.g., [np.arange(12) + 1])
        avg_stats :     average statistics for all levels True/False (default=True)
        percentile :    percentil for splitting the dataset in small and high intensities (default=50)
        
        """
        if months is None:
            months = [np.arange(12) + 1]

        self.precip.months = months
        self.precip.stats = melodist.build_casc(self.data, months=months, avg_stats=avg_stats, percentile=percentile)