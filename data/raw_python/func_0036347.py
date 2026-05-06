def disconnect_all(self):
        '''Disconnect all connections to this port.'''
        with self._mutex:
            for conn in self.connections:
                self.object.disconnect(conn.id)
            self.reparse_connections()