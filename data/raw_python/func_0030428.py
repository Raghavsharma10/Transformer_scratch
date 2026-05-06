def asDatetime(self, tzinfo=None):
        """Return this time as an aware datetime.datetime instance.

        The returned datetime object has the specified tzinfo, or a tzinfo
        describing UTC if the tzinfo parameter is None.
        """
        if tzinfo is None:
            tzinfo = FixedOffset(0, 0)

        if not self.isTimezoneDependent():
            return self._time.replace(tzinfo=tzinfo)
        else:
            return self._time.replace(tzinfo=FixedOffset(0, 0)).astimezone(tzinfo)