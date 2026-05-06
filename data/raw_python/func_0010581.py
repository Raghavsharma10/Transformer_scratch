def get_cqz(self, callsign, timestamp=timestamp_now):
        """ Returns CQ Zone of a callsign

        Args:
            callsign (str): Amateur Radio callsign
            timestamp (datetime, optional): datetime in UTC (tzinfo=pytz.UTC)

        Returns:
            int: containing the callsign's CQ Zone

        Raises:
            KeyError: no CQ Zone found for callsign

        """
        return self.get_all(callsign, timestamp)[const.CQZ]