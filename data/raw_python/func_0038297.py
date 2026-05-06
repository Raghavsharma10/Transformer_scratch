def _connect(self, config):
        """Establish a connection with a MySQL database."""
        if 'connection_timeout' not in self._config:
            self._config['connection_timeout'] = 480
        try:
            self._cnx = connect(**config)
            self._cursor = self._cnx.cursor()
            self._printer('\tMySQL DB connection established with db', config['database'])
        except Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            raise err