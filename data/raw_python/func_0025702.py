def remove(self, sid):
        """
        Remove sensor from the database

        Parameters
        ----------
        sid : str
            SensorID
        """
        self.dbcur.execute(SQL_SENSOR_DEL, (sid,))
        self.dbcur.execute(SQL_TMPO_DEL, (sid,))