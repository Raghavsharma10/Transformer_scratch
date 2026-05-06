def fromPOSIXTimestamp(klass, secs):
        """Return a new Time instance from seconds since the POSIX epoch.

        The POSIX epoch is midnight Jan 1, 1970 UTC. According to POSIX, leap
        seconds don't exist, so one UTC day is exactly 86400 seconds, even if
        it wasn't.

        @param secs: a number of seconds, represented as an integer, long or
        float.
        """
        self = klass.fromDatetime(_EPOCH + datetime.timedelta(seconds=secs))
        self.resolution = datetime.timedelta()
        return self