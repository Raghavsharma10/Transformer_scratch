def _build_key_patterns(self, slug, date):
        """Builds an OrderedDict of metric keys and patterns for the given slug
        and date."""
        # we want to keep the order, from smallest to largest granularity
        patts = OrderedDict()
        metric_key_patterns = self._metric_key_patterns()
        for g in self._granularities():
            date_string = date.strftime(metric_key_patterns[g]["date_format"])
            patts[g] = metric_key_patterns[g]["key"].format(slug, date_string)
        return patts