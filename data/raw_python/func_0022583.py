def _granularities(self):
        """Returns a generator of all possible granularities based on the
        MIN_GRANULARITY and MAX_GRANULARITY settings.
        """
        keep = False
        for g in GRANULARITIES:
            if g == app_settings.MIN_GRANULARITY and not keep:
                keep = True
            elif g == app_settings.MAX_GRANULARITY and keep:
                keep = False
                yield g
            if keep:
                yield g