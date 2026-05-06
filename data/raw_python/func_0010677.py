def lookup_zone_exception(self, callsign, timestamp=datetime.utcnow().replace(tzinfo=UTC)):
        """
        Returns a CQ Zone if an exception exists for the given callsign

        Args:
        callsign (string): Amateur radio callsign
        timestamp (datetime, optional): datetime in UTC (tzinfo=pytz.UTC)

        Returns:
            int: Value of the the CQ Zone exception which exists for this callsign (at the given time)

        Raises:
            KeyError: No matching callsign found
            APIKeyMissingError: API Key for Clublog missing or incorrect

        Example:
           The following code checks the Clublog XML database if a CQ Zone exception exists for the callsign DP0GVN.

           >>> from pyhamtools import LookupLib
           >>> my_lookuplib = LookupLib(lookuptype="clublogxml", apikey="myapikey")
           >>> print my_lookuplib.lookup_zone_exception("DP0GVN")
           38

           The prefix "DP" It is assigned to Germany, but the station is located in Antarctica, and therefore
           in CQ Zone 38

        Note:
            This method is available for

            - clublogxml
            - redis

        """

        callsign = callsign.strip().upper()

        if self._lookuptype == "clublogxml":

            return self._check_zone_exception_for_date(callsign, timestamp, self._zone_exceptions, self._zone_exceptions_index)

        elif self._lookuptype == "redis":

            data_dict, index = self._get_dicts_from_redis("_zone_ex_", "_zone_ex_index_", self._redis_prefix, callsign)
            return self._check_zone_exception_for_date(callsign, timestamp, data_dict, index)

        #no matching case
        raise KeyError