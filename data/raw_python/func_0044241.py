def hebrew_date(self):
        """Return the hebrew date string."""
        return u"{} {} {}".format(
            hebrew_number(self.hdate.day, hebrew=self.hebrew),   # Day
            htables.MONTHS[self.hdate.month - 1][self.hebrew],   # Month
            hebrew_number(self.hdate.year, hebrew=self.hebrew))