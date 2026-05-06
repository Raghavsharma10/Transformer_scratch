def getBounds(self, tzinfo=None):
        """
        Return a pair describing the bounds of self.

        This returns a pair (min, max) of Time instances. It is not quite the
        same as (self, self + self.resolution). This is because timezones are
        insignificant for instances with a resolution greater or equal to 1
        day.

        To illustrate the problem, consider a Time instance::

            T = Time.fromHumanly('today', tzinfo=anything)

        This will return an equivalent instance independent of the tzinfo used.
        The hour, minute, and second of this instance are 0, and its resolution
        is one day.

        Now say we have a sorted list of times, and we want to get all times
        for 'today', where whoever said 'today' is in a timezone that's 5 hours
        ahead of UTC. The start of 'today' in this timezone is UTC 05:00. The
        example instance T above is before this, but obviously it is today.

        The min and max times this returns are such that all potentially
        matching instances are within this range. However, this range might
        contain unmatching instances.

        As an example of this, if 'today' is April first 2005, then
        Time.fromISO8601TimeAndDate('2005-04-01T00:00:00') sorts in the same
        place as T from above, but is not in the UTC+5 'today'.

        TIME IS FUN!
        """
        if self.resolution >= datetime.timedelta(days=1) \
        and tzinfo is not None:
            time = self._time.replace(tzinfo=tzinfo)
        else:
            time = self._time

        return (
            min(self.fromDatetime(time), self.fromDatetime(self._time)),
            max(self.fromDatetime(time + self.resolution),
                self.fromDatetime(self._time + self.resolution))
        )