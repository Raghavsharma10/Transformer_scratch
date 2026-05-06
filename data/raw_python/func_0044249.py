def upcoming_shabbat(self):
        """Return the HDate for either the upcoming or current Shabbat.

        If it is currently Shabbat, returns the HDate of the Saturday.
        """
        if self.is_shabbat:
            return self
        # If it's Sunday, fast forward to the next Shabbat.
        saturday = self.gdate + datetime.timedelta(
            (12 - self.gdate.weekday()) % 7)
        return HDate(saturday, diaspora=self.diaspora, hebrew=self.hebrew)