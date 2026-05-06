def remove(self, connection):
        '''Remove a connection'''
        key = (connection.host, connection.port)
        with self._lock:
            found = self._connections.pop(key, None)
        try:
            self.close_connection(found)
        except Exception as exc:
            logger.warn('Failed to close %s: %s', connection, exc)
        return found