def previous_day(self):
        """Return the HDate for the previous day."""
        return HDate(self.gdate + datetime.timedelta(-1), self.diaspora,
                     self.hebrew)