def get_country_name(self, callsign, timestamp=timestamp_now):
        """ Returns the country name where the callsign is located

        Args:
            callsign (str): Amateur Radio callsign
            timestamp (datetime, optional): datetime in UTC (tzinfo=pytz.UTC)

        Returns:
            str: name of the Country

        Raises:
            KeyError: No Country found for callsign

        Note:
            Don't rely on the country name when working with several instances of
            py:class:`Callinfo`. Clublog and Country-files.org use slightly different names
            for countries. Example:

            - Country-files.com: "Fed. Rep. of Germany"
            - Clublog: "FEDERAL REPUBLIC OF GERMANY"

        """
        return self.get_all(callsign, timestamp)[const.COUNTRY]