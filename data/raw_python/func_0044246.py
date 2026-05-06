def omer_day(self):
        """Return the day of the Omer."""
        first_omer_day = HebrewDate(self.hdate.year, Months.Nisan, 16)
        omer_day = self._jdn - conv.hdate_to_jdn(first_omer_day) + 1
        if not 0 < omer_day < 50:
            return 0
        return omer_day