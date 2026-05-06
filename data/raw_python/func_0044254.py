def upcoming_yom_tov(self):
        """Find the next upcoming yom tov (i.e. no-melacha holiday).

        If it is currently the day of yom tov (irrespective of zmanim), returns
        that yom tov.
        """
        if self.is_yom_tov:
            return self
        this_year = self.get_holidays_for_year([HolidayTypes.YOM_TOV])
        next_rosh_hashana = HDate(
            heb_date=HebrewDate(self.hdate.year + 1, Months.Tishrei, 1),
            diaspora=self.diaspora,
            hebrew=self.hebrew)
        next_year = next_rosh_hashana.get_holidays_for_year(
            [HolidayTypes.YOM_TOV])

        # Filter anything that's past.
        holidays_list = [
            holiday_hdate for _, holiday_hdate in chain(this_year, next_year)
            if holiday_hdate >= self
        ]

        holidays_list.sort(key=lambda h: h.gdate)

        return holidays_list[0]