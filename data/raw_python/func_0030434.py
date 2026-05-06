def oneDay(self):
        """Return a Time instance representing the day of the start of self.

        The returned new instance will be set to midnight of the day containing
        the first instant of self in the specified timezone, and have a
        resolution of datetime.timedelta(days=1).
        """
        day = self.__class__.fromDatetime(self.asDatetime().replace(
                hour=0, minute=0, second=0, microsecond=0))
        day.resolution = datetime.timedelta(days=1)
        return day