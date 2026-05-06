def add(self, connection):
        '''Add a connection'''
        key = (connection.host, connection.port)
        with self._lock:
            if key not in self._connections:
                self._connections[key] = connection
                self.added(connection)
                return connection
            else:
                return None