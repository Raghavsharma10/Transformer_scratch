def get_lat_long(self, callsign, timestamp=timestamp_now):
        """ Returns Latitude and Longitude for a callsign

        Args:
            callsign (str): Amateur Radio callsign
            timestamp (datetime, optional): datetime in UTC (tzinfo=pytz.UTC)

        Returns:
            dict: Containing Latitude and Longitude

        Raises:
            KeyError: No data found for callsign

        Example:
            The following code returns Latitude & Longitude for "DH1TW"

            >>> from pyhamtools import LookupLib, Callinfo
            >>> my_lookuplib = LookupLib(lookuptype="countryfile")
            >>> cic = Callinfo(my_lookuplib)
            >>> cic.get_lat_long("DH1TW")
            {
                'latitude': 51.0,
                'longitude': -10.0
            }

        Note:
            Unfortunately, in most cases the returned Latitude and Longitude are not very precise.
            Clublog and Country-files.com use the country's capital coordinates in most cases, if no
            dedicated entry in the database exists. Best results will be retrieved with QRZ.com Lookup.

        """
        callsign_data = self.get_all(callsign, timestamp=timestamp)
        return {
            const.LATITUDE: callsign_data[const.LATITUDE],
            const.LONGITUDE: callsign_data[const.LONGITUDE]
        }