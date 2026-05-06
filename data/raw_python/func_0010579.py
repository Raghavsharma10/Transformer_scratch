def is_valid_callsign(self, callsign, timestamp=timestamp_now):
        """ Checks if a callsign is valid

        Args:
            callsign (str): Amateur Radio callsign
            timestamp (datetime, optional): datetime in UTC (tzinfo=pytz.UTC)

        Returns:
            bool: True / False

        Example:
            The following checks if "DH1TW" is a valid callsign

            >>> from pyhamtools import LookupLib, Callinfo
            >>> my_lookuplib = LookupLib(lookuptype="countryfile")
            >>> cic = Callinfo(my_lookuplib)
            >>> cic.is_valid_callsign("DH1TW")
            True

        """
        try:
            if self.get_all(callsign, timestamp):
                return True
        except KeyError:
            return False