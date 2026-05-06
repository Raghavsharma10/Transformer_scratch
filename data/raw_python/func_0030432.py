def asHumanly(self, tzinfo=None, now=None, precision=Precision.MINUTES):
        """Return this time as a short string, tailored to the current time.

        Parts of the date that can be assumed are omitted. Consequently, the
        output string depends on the current time. This is the format used for
        displaying dates in most user visible places in the quotient web UI.

        By default, the current time is determined by the system clock. The
        current time used for formatting the time can be changed by providing a
        Time instance as the parameter 'now'.

        @param precision: The smallest unit of time that will be represented
        in the returned string.  Valid values are L{Time.Precision.MINUTES} and
        L{Time.Precision.SECONDS}.

        @raise InvalidPrecision: if the specified precision is not either
        L{Time.Precision.MINUTES} or L{Time.Precision.SECONDS}.
        """
        try:
            timeFormat = Time._timeFormat[precision]
        except KeyError:
            raise InvalidPrecision(
                    'Use Time.Precision.MINUTES or Time.Precision.SECONDS')

        if now is None:
            now = Time().asDatetime(tzinfo)
        else:
            now = now.asDatetime(tzinfo)
        dtime = self.asDatetime(tzinfo)

        # Same day?
        if dtime.date() == now.date():
            if self.isAllDay():
                return 'all day'
            return dtime.strftime(timeFormat).lower()
        else:
            res = str(dtime.date().day) + dtime.strftime(' %b')  # day + month
            # Different year?
            if not dtime.date().year == now.date().year:
                res += dtime.strftime(' %Y')
            if not self.isAllDay():
                res += dtime.strftime(', %s' % (timeFormat,)).lower()
            return res