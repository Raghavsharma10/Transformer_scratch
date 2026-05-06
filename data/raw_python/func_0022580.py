def _date_range(self, granularity, since, to=None):
        """Returns a generator that yields ``datetime.datetime`` objects from
        the ``since`` date until ``to`` (default: *now*).

        * ``granularity`` -- The granularity at which the generated datetime
          objects should be created: seconds, minutes, hourly, daily, weekly,
          monthly, or yearly
        * ``since`` -- a ``datetime.datetime`` object, from which we start
          generating periods of time. This can also be ``None``, and will
          default to the past 7 days if that's the case.
        * ``to`` -- a ``datetime.datetime`` object, from which we start
          generating periods of time. This can also be ``None``, and will
          default to now if that's the case.

        If ``granularity`` is one of daily, weekly, monthly, or yearly, this
        function gives objects at the daily level.

        If ``granularity`` is one of the following, the number of datetime
        objects returned is capped, otherwise this code is really slow and
        probably generates more data than we want:

            * hourly: returns at most 720 values (~30 days)
            * minutes: returns at most 480 values (8 hours)
            * second: returns at most 300 values (5 minutes)

        For example, if granularity is "seconds", we'll receive datetime
        objects that differ by 1 second each.

        """
        if since is None:
            since = datetime.utcnow() - timedelta(days=7)  # Default to 7 days

        if to is None:
            to = datetime.utcnow()
        elapsed = (to - since)

        # Figure out how many units to generate for the elapsed time.
        # I'm going to use `granularity` as a keyword parameter to timedelta,
        # so I need to change the wording for hours and anything > days.
        if granularity == "seconds":
            units = elapsed.total_seconds()
            units = 300 if units > 300 else units
        elif granularity == "minutes":
            units = elapsed.total_seconds() / 60
            units = 480 if units > 480 else units
        elif granularity == "hourly":
            granularity = "hours"
            units = elapsed.total_seconds() / 3600
            units = 720 if units > 720 else units
        else:
            granularity = "days"
            units = elapsed.days + 1

        return (to - timedelta(**{granularity: u}) for u in range(int(units)))