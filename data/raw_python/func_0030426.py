def fromDatetime(klass, dtime):
        """Return a new Time instance from a datetime.datetime instance.

        If the datetime instance does not have an associated timezone, it is
        assumed to be UTC.
        """
        self = klass.__new__(klass)
        if dtime.tzinfo is not None:
            self._time = dtime.astimezone(FixedOffset(0, 0)).replace(tzinfo=None)
        else:
            self._time = dtime
        self.resolution = datetime.timedelta.resolution
        return self