def utc_minute_timezone(self, minutes_from_utc):
        """Return the local time for a given time UTC."""
        from_zone = tz.gettz('UTC')
        to_zone = self.location.timezone
        utc = dt.datetime.combine(self.date, dt.time()) + \
            dt.timedelta(minutes=minutes_from_utc)
        utc = utc.replace(tzinfo=from_zone)
        local = utc.astimezone(to_zone)
        return local