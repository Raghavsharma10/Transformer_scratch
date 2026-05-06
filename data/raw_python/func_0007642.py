def interval(self, granularity):
        """
        Note that if you don't specify a granularity (either through the `interval`
        method or through the `hourly`, `daily`, `weekly`, `monthly` or `yearly`
        shortcut methods) you will get only a single result, encompassing the
        entire date range, per metric.
        """

        if granularity == 'total':
            return self

        if not isinstance(granularity, int):
            if granularity in self.GRANULARITY_LEVELS:
                granularity = self.GRANULARITY_LEVELS.index(granularity)
            else:
                levels = ", ".join(self.GRANULARITY_LEVELS)
                raise ValueError("Granularity should be one of: lifetime, " + levels)

        dimension = self.GRANULARITY_DIMENSIONS[granularity]
        self.raw['dimensions'].insert(0, dimension)

        return self