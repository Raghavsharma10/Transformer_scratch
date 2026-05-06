def asRFC2822(self, tzinfo=None, includeDayOfWeek=True):
        """Return this Time formatted as specified in RFC 2822.

        RFC 2822 specifies the format of email messages.

        RFC 2822 says times in email addresses should reflect the local
        timezone. If tzinfo is a datetime.tzinfo instance, the returned
        formatted string will reflect that timezone. Otherwise, the timezone
        will be '-0000', which RFC 2822 defines as UTC, but with an unknown
        local timezone.

        RFC 2822 states that the weekday is optional. The parameter
        includeDayOfWeek indicates whether or not to include it.
        """
        dtime = self.asDatetime(tzinfo)

        if tzinfo is None:
            rfcoffset = '-0000'
        else:
            rfcoffset = '%s%02i%02i' % _timedeltaToSignHrMin(dtime.utcoffset())

        rfcstring = ''
        if includeDayOfWeek:
            rfcstring += self.rfc2822Weekdays[dtime.weekday()] + ', '

        rfcstring += '%i %s %4i %02i:%02i:%02i %s' % (
            dtime.day,
            self.rfc2822Months[dtime.month - 1],
            dtime.year,
            dtime.hour,
            dtime.minute,
            dtime.second,
            rfcoffset)

        return rfcstring