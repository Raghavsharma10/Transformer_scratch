def add(self, sid, token):
        """
        Add new sensor to the database

        Parameters
        ----------
        sid : str
            SensorId
        token : str
        """
        try:
            self.dbcur.execute(SQL_SENSOR_INS, (sid, token))
        except sqlite3.IntegrityError:  # sensor entry exists
            pass