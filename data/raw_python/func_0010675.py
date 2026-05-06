def is_invalid_operation(self, callsign, timestamp=datetime.utcnow().replace(tzinfo=UTC)):
        """
        Returns True if an operations is known as invalid

        Args:
            callsign (string): Amateur Radio callsign
            timestamp (datetime, optional): datetime in UTC (tzinfo=pytz.UTC)

        Returns:
            bool: True if a record exists for this callsign (at the given time)

        Raises:
            KeyError: No matching callsign found
            APIKeyMissingError: API Key for Clublog missing or incorrect

        Example:
           The following code checks the Clublog XML database if the operation is valid for two dates.

           >>> from pyhamtools import LookupLib
           >>> from datetime import datetime
           >>> import pytz
           >>> my_lookuplib = LookupLib(lookuptype="clublogxml", apikey="myapikey")
           >>> print my_lookuplib.is_invalid_operation("5W1CFN")
           True
           >>> try:
           >>>   timestamp = datetime(year=2012, month=1, day=31).replace(tzinfo=pytz.UTC)
           >>>   my_lookuplib.is_invalid_operation("5W1CFN", timestamp)
           >>> except KeyError:
           >>>   print "Seems to be invalid operation before 31.1.2012"
           Seems to be an invalid operation before 31.1.2012

        Note:
            This method is available for

            - clublogxml
            - redis

        """

        callsign = callsign.strip().upper()

        if self._lookuptype == "clublogxml":

            return self._check_inv_operation_for_date(callsign, timestamp, self._invalid_operations, self._invalid_operations_index)

        elif self._lookuptype == "redis":

            data_dict, index = self._get_dicts_from_redis("_inv_op_", "_inv_op_index_", self._redis_prefix, callsign)
            return self._check_inv_operation_for_date(callsign, timestamp, data_dict, index)

        #no matching case
        raise KeyError