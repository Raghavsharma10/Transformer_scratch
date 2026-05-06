def connect(self):
        """ Connects to the redis database. """
        self._connection = StrictRedis(
            host=self._host,
            port=self._port,
            db=self._database,
            password=self._password)