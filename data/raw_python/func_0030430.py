def asISO8601TimeAndDate(self, includeDelimiters=True, tzinfo=None,
                             includeTimezone=True):
        """Return this time formatted as specified by ISO 8861.

        ISO 8601 allows optional dashes to delimit dates and colons to delimit
        times. The parameter includeDelimiters (default True) defines the
        inclusion of these delimiters in the output.

        If tzinfo is a datetime.tzinfo instance, the output time will be in the
        timezone given. If it is None (the default), then the timezone string
        will not be included in the output, and the time will be in UTC.

        The includeTimezone parameter coresponds to the inclusion of an
        explicit timezone. The default is True.
        """
        if not self.isTimezoneDependent():
            tzinfo = None
        dtime = self.asDatetime(tzinfo)

        if includeDelimiters:
            dateSep = '-'
            timeSep = ':'
        else:
            dateSep = timeSep = ''

        if includeTimezone:
            if tzinfo is None:
                timezone = '+00%s00' % (timeSep,)
            else:
                sign, hour, min = _timedeltaToSignHrMin(dtime.utcoffset())
                timezone = '%s%02i%s%02i' % (sign, hour, timeSep, min)
        else:
            timezone = ''

        microsecond = ('%06i' % (dtime.microsecond,)).rstrip('0')
        if microsecond:
            microsecond = '.' + microsecond

        parts = [
            ('%04i' % (dtime.year,), datetime.timedelta(days=366)),
            ('%s%02i' % (dateSep, dtime.month), datetime.timedelta(days=31)),
            ('%s%02i' % (dateSep, dtime.day), datetime.timedelta(days=1)),
            ('T', datetime.timedelta(hours=1)),
            ('%02i' % (dtime.hour,), datetime.timedelta(hours=1)),
            ('%s%02i' % (timeSep, dtime.minute), datetime.timedelta(minutes=1)),
            ('%s%02i' % (timeSep, dtime.second), datetime.timedelta(seconds=1)),
            (microsecond, datetime.timedelta(microseconds=1)),
            (timezone, datetime.timedelta(hours=1))
        ]

        formatted = ''
        for part, minResolution in parts:
            if self.resolution <= minResolution:
                formatted += part

        return formatted