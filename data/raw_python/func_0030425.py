def fromStructTime(klass, structTime, tzinfo=None):
        """Return a new Time instance from a time.struct_time.

        If tzinfo is None, structTime is in UTC. Otherwise, tzinfo is a
        datetime.tzinfo instance coresponding to the timezone in which
        structTime is.

        Many of the functions in the standard time module return these things.
        This will also work with a plain 9-tuple, for parity with the time
        module. The last three elements, or tm_wday, tm_yday, and tm_isdst are
        ignored.
        """
        dtime = datetime.datetime(tzinfo=tzinfo, *structTime[:6])
        self = klass.fromDatetime(dtime)
        self.resolution = datetime.timedelta(seconds=1)
        return self