def post_build_time_coverage(self):
        """Collect all of the time coverage for the bundle."""
        from ambry.util.datestimes import expand_to_years

        years = set()

        # From the bundle about
        if self.metadata.about.time:
            for year in expand_to_years(self.metadata.about.time):
                years.add(year)

        # From the bundle name
        if self.identity.btime:
            for year in expand_to_years(self.identity.btime):
                years.add(year)

        # From all of the partitions
        for p in self.partitions:
            years |= set(p.time_coverage)