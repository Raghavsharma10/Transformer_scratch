def get_adif_id(self, callsign, timestamp=timestamp_now):
        """ Returns ADIF id of a callsign's country

        Args:
            callsign (str): Amateur Radio callsign
            timestamp (datetime, optional): datetime in UTC (tzinfo=pytz.UTC)

        Returns:
            int: containing the country ADIF id

        Raises:
            KeyError: No Country found for callsign

        """
        return self.get_all(callsign, timestamp)[const.ADIF]