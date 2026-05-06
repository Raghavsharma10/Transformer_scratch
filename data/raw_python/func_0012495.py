def _generate_badges(self):
        """
        Generate download badges. Append them to ``self._badges``.
        """
        daycount = self._stats.downloads_per_day
        day = self._generate_badge('Downloads', '%d/day' % daycount)
        self._badges['per-day'] = day
        weekcount = self._stats.downloads_per_week
        if weekcount is None:
            # we don't have enough data for week (or month)
            return
        week = self._generate_badge('Downloads', '%d/week' % weekcount)
        self._badges['per-week'] = week
        monthcount = self._stats.downloads_per_month
        if monthcount is None:
            # we don't have enough data for month
            return
        month = self._generate_badge('Downloads', '%d/month' % monthcount)
        self._badges['per-month'] = month