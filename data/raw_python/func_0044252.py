def last_day(self):
        """Return the last day of Yom Tov or Shabbat.

        This is useful for three-day holidays, for example: it will return the
        last in a string of Yom Tov + Shabbat.
        If this HDate is Shabbat followed by no Yom Tov, returns the Saturday.
        If this HDate is neither Yom Tov, nor Shabbat, this just returns
        itself.
        """
        day_iter = self
        while day_iter.next_day.is_yom_tov or day_iter.next_day.is_shabbat:
            day_iter = day_iter.next_day
        return day_iter