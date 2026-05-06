def get_connection_by_dest(self, dest):
        '''DEPRECATED. Search for a connection between this and another port.'''
        with self._mutex:
            for conn in self.connections:
                if conn.has_port(self) and conn.has_port(dest):
                    return conn
            return None