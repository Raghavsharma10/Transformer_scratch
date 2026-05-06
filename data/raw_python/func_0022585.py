def _build_keys(self, slug, date=None, granularity='all'):
        """Builds redis keys used to store metrics.

        * ``slug`` -- a slug used for a metric, e.g. "user-signups"
        * ``date`` -- (optional) A ``datetime.datetime`` object used to
          generate the time period for the metric. If omitted, the current date
          and time (in UTC) will be used.
        * ``granularity`` -- Must be one of: "all" (default), "yearly",
        "monthly", "weekly", "daily", "hourly", "minutes", or "seconds".

        Returns a list of strings.

        """
        slug = slugify(slug)  # Ensure slugs have a consistent format
        if date is None:
            date = datetime.utcnow()
        patts = self._build_key_patterns(slug, date)
        if granularity == "all":
            return list(patts.values())
        return [patts[granularity]]