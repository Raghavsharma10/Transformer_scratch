def next_day(self):
        """Return the HDate for the next day."""
        return HDate(self.gdate + datetime.timedelta(1), self.diaspora,
                     self.hebrew)