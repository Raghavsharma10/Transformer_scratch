def last_timestamp(self, sid, epoch=False):
        """
        Get the theoretical last timestamp for a sensor

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
        timestamp, value = self.last_datapoint(sid, epoch)
        return timestamp