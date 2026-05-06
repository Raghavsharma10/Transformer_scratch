def purge(self, connection):
        """Remove the connection from rotation"""
        self._checkpid()
        if connection.pid == self.pid:
            idx = connection._pattern_idx
            if connection in self._in_use_connections[idx]:
                self._in_use_connections[idx].remove(connection)
            else:
                self._available_connections[idx].remove(connection)
            connection.disconnect()