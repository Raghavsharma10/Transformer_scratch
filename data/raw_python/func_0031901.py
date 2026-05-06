def get(self):
        """Return the computed statistics over the gathered data"""

        values = self.reservoir.sorted_values

        def safe(f, *args):
            try:
                return f(values, *args)
            except exceptions.StatisticsError:
                return 0.0

        plevels = [50, 75, 90, 95, 99, 99.9]
        percentiles = [safe(statistics.percentile, p) for p in plevels]

        try:
            histogram = statistics.get_histogram(values)
        except exceptions.StatisticsError:
            histogram = [(0, 0)]

        res = dict(
            kind="histogram",
            min=values[0] if values else 0,
            max=values[-1] if values else 0,
            arithmetic_mean=safe(statistics.mean),
            geometric_mean=safe(statistics.geometric_mean),
            harmonic_mean=safe(statistics.harmonic_mean),
            median=safe(statistics.median),
            variance=safe(statistics.variance),
            standard_deviation=safe(statistics.stdev),
            skewness=safe(statistics.skewness),
            kurtosis=safe(statistics.kurtosis),
            percentile=py3comp.zip(plevels, percentiles),
            histogram=histogram,
            n=len(values))
        return res