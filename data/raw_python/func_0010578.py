def get_all(self, callsign, timestamp=timestamp_now):
        """ Lookup a callsign and return all data available from the underlying database

        Args:
            callsign (str): Amateur Radio callsign
            timestamp (datetime, optional): datetime in UTC (tzinfo=pytz.UTC)

        Returns:
            dict: Dictionary containing the callsign specific data

        Raises:
            KeyError: Callsign could not be identified

        Example:
            The following code returns all available information from the country-files.com database for the
            callsign "DH1TW"

            >>> from pyhamtools import LookupLib, Callinfo
            >>> my_lookuplib = LookupLib(lookuptype="countryfile")
            >>> cic = Callinfo(my_lookuplib)
            >>> cic.get_all("DH1TW")
            {
                'country': 'Fed. Rep. of Germany',
                'adif': 230,
                'continent': 'EU',
                'latitude': 51.0,
                'longitude': -10.0,
                'cqz': 14,
                'ituz': 28
            }

        Note:
            The content of the returned data depends entirely on the injected
            :py:class:`LookupLib` (and the used database). While the country-files.com provides
            for example the ITU Zone, Clublog doesn't. Consequently, the item "ituz"
            would be missing with Clublog (API or XML) :py:class:`LookupLib`.

        """
        callsign_data = self._lookup_callsign(callsign, timestamp)

        try:
            cqz = self._lookuplib.lookup_zone_exception(callsign, timestamp)
            callsign_data[const.CQZ] = cqz
        except KeyError:
            pass

        return callsign_data