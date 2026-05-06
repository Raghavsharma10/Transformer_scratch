def hdate(self, date):
        """Set the dates of the HDate object based on a given Hebrew date."""
        # Sanity checks
        if date is None and isinstance(self.gdate, datetime.date):
            # Calculate the value since gdate has been set
            date = self.hdate

        if not isinstance(date, HebrewDate):
            raise TypeError('date: {} is not of type HebrewDate'.format(date))
        if not 0 < date.month < 15:
            raise ValueError(
                'month ({}) legal values are 1-14'.format(date.month))
        if not 0 < date.day < 31:
            raise ValueError('day ({}) legal values are 1-31'.format(date.day))

        self._last_updated = "hdate"
        self._hdate = date