def stats(self):
        """ -> :class:collections.OrderedDict of stats about the time intervals
        """
        return OrderedDict([
            ("Intervals", len(self.array)),
            ("Mean", self.format_time(self.mean or 0)),
            ("Min", self.format_time(self.min or 0)),
            ("Median", self.format_time(self.median or 0)),
            ("Max", self.format_time(self.max or 0)),
            ("St. Dev.", self.format_time(self.stdev or 0)),
            ("Total", self.format_time(self.exectime or 0)),
        ])