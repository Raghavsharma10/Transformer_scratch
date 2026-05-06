def last_datapoint(self, sid, epoch=False):
        """
        Parameters
        ----------
        sid : str
            SensorId
        epoch : bool
            default False
            If True return as epoch
            If False return as pd.Timestamp

        Returns
        -------
        pd.Timestamp | int, float
        """
        block = self._last_block(sid)
        if block is None:
            return None, None

        header = block['h']
        timestamp, value = header['tail']

        if not epoch:
            timestamp = pd.Timestamp.utcfromtimestamp(timestamp)
            timestamp = timestamp.tz_localize('UTC')

        return timestamp, value