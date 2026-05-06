def get_connection_by_id(self, id):
        '''Search for a connection on this port by its ID.'''
        with self._mutex:
            for conn in self.connections:
                if conn.id == id:
                    return conn
            return None