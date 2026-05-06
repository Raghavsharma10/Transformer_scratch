def _connect(self):
        """Establish connection to PostgreSQL Database."""
        if self._connParams:
            self._conn = psycopg2.connect(**self._connParams)
        else:
            self._conn = psycopg2.connect('')
        try:
            ver_str = self._conn.get_parameter_status('server_version')
        except AttributeError:
            ver_str = self.getParam('server_version')
        self._version = util.SoftwareVersion(ver_str)