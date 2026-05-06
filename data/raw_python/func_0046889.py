def _parse_last_build_date(self):
        """
        Returns the last build date of the RSS feed as datetime.datetime
        object. Returned datetime is not time-zone aware
        """
        date = self._channel.find('lastBuildDate').text
        date = parser.parse(date, ignoretz=True)
        return date