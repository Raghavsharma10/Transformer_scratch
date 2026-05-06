def get_continent(self, callsign, timestamp=timestamp_now):
        """ Returns the continent Identifier of a callsign

        Args:
            callsign (str): Amateur Radio callsign
            timestamp (datetime, optional): datetime in UTC (tzinfo=pytz.UTC)

        Returns:
            str: continent identified

        Raises:
            KeyError: No Continent found for callsign

        Note:
            The following continent identifiers are used:

            - EU: Europe
            - NA: North America
            - SA: South America
            - AS: Asia
            - AF: Africa
            - OC: Oceania
            - AN: Antarctica
        """
        return self.get_all(callsign, timestamp)[const.CONTINENT]