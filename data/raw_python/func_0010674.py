def lookup_prefix(self, prefix, timestamp=timestamp_now):
        """
        Returns lookup data of a Prefix

        Args:
            prefix (string): Prefix of a Amateur Radio callsign
            timestamp (datetime, optional): datetime in UTC (tzinfo=pytz.UTC)

        Returns:
            dict: Dictionary containing the country specific data of the Prefix

        Raises:
            KeyError: No matching Prefix found
            APIKeyMissingError: API Key for Clublog missing or incorrect

        Example:
           The following code shows how to obtain the information for the prefix "DH" from the countryfile.com
           database (default database).

           >>> from pyhamtools import LookupLib
           >>> myLookupLib = LookupLib()
           >>> print myLookupLib.lookup_prefix("DH")
           {
            'adif': 230,
            'country': u'Fed. Rep. of Germany',
            'longitude': 10.0,
            'cqz': 14,
            'ituz': 28,
            'latitude': 51.0,
            'continent': u'EU'
           }

        Note:
            This method is available for

            - clublogxml
            - countryfile
            - redis

        """

        prefix = prefix.strip().upper()

        if self._lookuptype == "clublogxml" or self._lookuptype == "countryfile":

            return self._check_data_for_date(prefix, timestamp, self._prefixes, self._prefixes_index)

        elif self._lookuptype == "redis":

            data_dict, index = self._get_dicts_from_redis("_prefix_", "_prefix_index_", self._redis_prefix, prefix)
            return self._check_data_for_date(prefix, timestamp, data_dict, index)

        # no matching case
        raise KeyError