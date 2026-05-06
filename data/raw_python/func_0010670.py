def lookup_callsign(self, callsign=None, timestamp=timestamp_now):
        """
        Returns lookup data if an exception exists for a callsign

        Args:
            callsign (string): Amateur radio callsign
            timestamp (datetime, optional): datetime in UTC (tzinfo=pytz.UTC)

        Returns:
            dict: Dictionary containing the country specific data of the callsign

        Raises:
            KeyError: No matching callsign found
            APIKeyMissingError: API Key for Clublog missing or incorrect

        Example:
           The following code queries the the online Clublog API for the callsign "VK9XO" on a specific date.

           >>> from pyhamtools import LookupLib
           >>> from datetime import datetime
           >>> import pytz
           >>> my_lookuplib = LookupLib(lookuptype="clublogapi", apikey="myapikey")
           >>> timestamp = datetime(year=1962, month=7, day=7, tzinfo=pytz.UTC)
           >>> print my_lookuplib.lookup_callsign("VK9XO", timestamp)
           {
            'country': u'CHRISTMAS ISLAND',
            'longitude': 105.7,
            'cqz': 29,
            'adif': 35,
            'latitude': -10.5,
            'continent': u'OC'
           }

        Note:
            This method is available for

            - clublogxml
            - clublogapi
            - countryfile
            - qrz.com
            - redis


        """
        callsign = callsign.strip().upper()

        if self._lookuptype == "clublogapi":
            callsign_data =  self._lookup_clublogAPI(callsign=callsign, timestamp=timestamp, apikey=self._apikey)
            if callsign_data[const.ADIF]==1000:
                raise KeyError
            else:
                return callsign_data

        elif self._lookuptype == "clublogxml" or self._lookuptype == "countryfile":

            return self._check_data_for_date(callsign, timestamp, self._callsign_exceptions, self._callsign_exceptions_index)

        elif self._lookuptype == "redis":

            data_dict, index = self._get_dicts_from_redis("_call_ex_", "_call_ex_index_", self._redis_prefix, callsign)
            return self._check_data_for_date(callsign, timestamp, data_dict, index)

        # no matching case
        elif self._lookuptype == "qrz":
            return self._lookup_qrz_callsign(callsign, self._apikey, self._apiv)

        raise KeyError("unknown Callsign")