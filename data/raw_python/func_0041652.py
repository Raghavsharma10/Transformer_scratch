def locked(self):
        """Context generator for `with` statement, yields thread-safe connection.

        :return: thread-safe connection
        :rtype: pydbal.connection.Connection
        """
        conn = self._get_connection()
        try:
            self._lock(conn)
            yield conn
        finally:
            self._unlock(conn)