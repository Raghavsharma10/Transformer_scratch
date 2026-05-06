def _get_connection(self):
        """ Returns connection to the postgres database.

        Returns:
            connection to postgres database who stores mpr data.

        """
        if not getattr(self, '_connection', None):
            logger.debug(
                'Creating new connection.\n   dsn: {}'
                .format(self._dsn))

            d = parse_url_to_dict(self._dsn)
            self._connection = psycopg2.connect(
                database=d['path'].strip('/'), user=d['username'], password=d['password'],
                port=d['port'], host=d['hostname'])
            # It takes some time to find the way how to get raw connection from sqlalchemy. So,
            # I leave the commented code.
            #
            # self._engine = create_engine(self._dsn)
            # self._connection = self._engine.raw_connection()
            #
        return self._connection