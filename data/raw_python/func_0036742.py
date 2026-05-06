def _parse_datetime_str(self, dtime_str):
        """
        Given a standard datetime string (as seen throughout the Petfinder API),
        spit out the corresponding UTC datetime instance.

        :param str dtime_str: The datetime string to parse.
        :rtype: datetime.datetime
        :returns: The parsed datetime.
        """

        return datetime.datetime.strptime(
            dtime_str,
            "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=pytz.utc)