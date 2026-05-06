def _get_connection(self):
        """ Returns connection to sqlite db.

        Returns:
            connection to the sqlite db who stores mpr data.

        """
        if getattr(self, '_connection', None):
            logger.debug('Connection to sqlite db already exists. Using existing one.')
        else:
            dsn = self._dsn
            if dsn == 'sqlite://':
                dsn = ':memory:'
            else:
                dsn = dsn.replace('sqlite:///', '')

            logger.debug(
                'Creating new apsw connection.\n   dsn: {}, config_dsn: {}'
                .format(dsn, self._dsn))
            self._connection = apsw.Connection(dsn)

        return self._connection