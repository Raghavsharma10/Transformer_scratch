def _execute(self, sql, params):
        """Execute statement with reconnecting by connection closed error codes.

        2006 (CR_SERVER_GONE_ERROR): MySQL server has gone away
        2013 (CR_SERVER_LOST): Lost connection to MySQL server during query
        2055 (CR_SERVER_LOST_EXTENDED): Lost connection to MySQL server at '%s', system error: %d
        """
        try:
            return self._execute_unsafe(sql, params)
        except MySQLdb.OperationalError as ex:
            if ex.args[0] in (2006, 2013, 2055):
                self._log("Connection with server is lost. Trying to reconnect.")
                self.connect()
                return self._execute_unsafe(sql, params)
            raise