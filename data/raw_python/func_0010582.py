def get_ituz(self, callsign, timestamp=timestamp_now):
        """ Returns ITU Zone of a callsign

        Args:
            callsign (str): Amateur Radio callsign
            timestamp (datetime, optional): datetime in UTC (tzinfo=pytz.UTC)

        Returns:
            int: containing the callsign's CQ Zone

        Raises:
            KeyError: No ITU Zone found for callsign

        Note:
            Currently, only Country-files.com lookup database contains ITU Zones

        """
        return self.get_all(callsign, timestamp)[const.ITUZ]