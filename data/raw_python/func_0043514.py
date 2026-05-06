def strptime(self, value, format):
        """
        By default, parse datetime with TZ.
        If TZ is False, convert datetime to local time and disable TZ
        """

        value = force_str(value)
        if format == ISO_8601:
            try:
                parsed = parse_datetime(value)
                if not settings.USE_TZ:
                        fr_tz = pytz.timezone(settings.TIME_ZONE)
                        parsed = parsed.astimezone(fr_tz).replace(tzinfo=None)
            except:
                raise APIException(
                    "date parsing error: since parameter use the date format ISO 8601 (ex: 2014-11-18T15:56:58Z)")

            if parsed is None:
                raise APIException(
                    "since parameter use the date format ISO 8601 (ex: 2014-11-18T15:56:58Z)")
            return parsed
        return super(IsoDateTimeField, self).strptime(value, format)