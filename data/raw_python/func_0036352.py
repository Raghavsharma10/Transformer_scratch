def get_connection_by_name(self, name):
        '''Search for a connection to or from this port by name.'''
        with self._mutex:
            for conn in self.connections:
                if conn.name == name:
                    return conn
            return None