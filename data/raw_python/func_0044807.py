def issur_melacha_in_effect(self):
        """At the given time, return whether issur melacha is in effect."""
        # TODO: Rewrite this in terms of candle_lighting/havdalah properties.
        weekday = self.date.weekday()
        tomorrow = self.date + dt.timedelta(days=1)
        tomorrow_holiday_type = HDate(
            gdate=tomorrow, diaspora=self.location.diaspora).holiday_type
        today_holiday_type = HDate(
            gdate=self.date, diaspora=self.location.diaspora).holiday_type

        if weekday == 4 or tomorrow_holiday_type == HolidayTypes.YOM_TOV:
            if self.time > (self.zmanim["sunset"] -
                            dt.timedelta(minutes=self.candle_lighting_offset)):
                return True
        if weekday == 5 or today_holiday_type == HolidayTypes.YOM_TOV:
            if self.time < self.zmanim["three_stars"]:
                return True
        return False