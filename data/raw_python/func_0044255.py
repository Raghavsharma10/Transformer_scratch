def get_reading(self):
        """Return number of hebrew parasha."""
        _year_type = (self.year_size() % 10) - 3
        year_type = (
            self.diaspora * 1000 +
            self.rosh_hashana_dow() * 100 +
            _year_type * 10 +
            self.pesach_dow())

        _LOGGER.debug("Year type: %d", year_type)

        # Number of days since rosh hashana
        rosh_hashana = HebrewDate(self.hdate.year, Months.Tishrei, 1)
        days = self._jdn - conv.hdate_to_jdn(rosh_hashana)
        # Number of weeks since rosh hashana
        weeks = (days + self.rosh_hashana_dow() - 1) // 7
        _LOGGER.debug("Days: %d, Weeks %d", days, weeks)

        # If it's currently Simchat Torah, return VeZot Haberacha.
        if weeks == 3:
            if (days <= 22 and self.diaspora and self.dow != 7 or
                    days <= 21 and not self.diaspora):
                return 54

        # Special case for Simchat Torah in diaspora.
        if weeks == 4 and days == 22 and self.diaspora:
            return 54

        # Return the indexes for the readings of the given year
        def unpack_readings(readings):
            return list(chain(
                *([x] if isinstance(x, int) else x for x in readings)))

        reading_for_year = htables.READINGS[year_type]
        readings = unpack_readings(reading_for_year)
        # Maybe recompute the year type based on the upcoming shabbat.
        # This avoids an edge case where today is before Rosh Hashana but
        # Shabbat is in a new year afterwards.
        if (weeks >= len(readings)
                and self.hdate.year < self.upcoming_shabbat.hdate.year):
            return self.upcoming_shabbat.get_reading()
        return readings[weeks]