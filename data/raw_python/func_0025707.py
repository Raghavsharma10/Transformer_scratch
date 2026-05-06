def first_timestamp(self, sid, epoch=False):
        """
        Get the first available timestamp for a sensor

        Parameters
        ----------
        sid : str
            SensorID
        epoch : bool
            default False
            If True return as epoch
            If False return as pd.Timestamp

        Returns
        -------
        pd.Timestamp | int
        """
        first_block = self.dbcur.execute(SQL_TMPO_FIRST, (sid,)).fetchone()
        if first_block is None:
            return None

        timestamp = first_block[2]
        if not epoch:
            timestamp = pd.Timestamp.utcfromtimestamp(timestamp)
            timestamp = timestamp.tz_localize('UTC')
        return timestamp