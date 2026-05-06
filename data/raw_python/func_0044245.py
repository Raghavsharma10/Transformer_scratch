def pesach_dow(self):
        """Return the first day of week for Pesach."""
        jdn = conv.hdate_to_jdn(HebrewDate(self.hdate.year, Months.Nisan, 15))
        return (jdn + 1) % 7 + 1