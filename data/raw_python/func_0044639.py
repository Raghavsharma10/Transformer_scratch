def normalize_date(date):
        '''normalize the specified date to milliseconds since the epoch

        If it is a string, it is assumed to be some sort of datetime such as
        "2015-12-27" or "2015-12-27T11:01:20.954". If date is a naive datetime,
        it is assumed to be UTC.

        If numeric arguments are beyond 5138-11-16 (100,000,000,000 seconds
        after epoch), they are interpreted as milliseconds since the epoch.
        '''

        if isinstance(date, datetime):
            pass
        elif date == "now":
            date = datetime.now(pytz.UTC)
        elif isinstance(date, (basestring, int, float, long)):
            try:
                ts = float(date)
                if ts > MAX_TS_SECONDS:
                    # ts was provided in ms
                    ts = ts / 1000.0
                # For unix timestamps on command line
                date = datetime.utcfromtimestamp(float(ts))
            except ValueError:
                try:
                    date = dateparse(date)
                except ValueError as e:
                    raise InvalidDatalakeMetadata(str(e))
        else:
            msg = 'could not parse a date from {!r}'.format(date)
            raise InvalidDatalakeMetadata(msg)

        return Metadata._from_datetime(date)