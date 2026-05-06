def get_connection(self, command_name, *keys, **options):
        """Get a connection from the pool"""
        self._checkpid()
        try:
            connection = self._available_connections[self._pattern_idx].pop()
        except IndexError:
            connection = self.make_connection()
        self._in_use_connections[self._pattern_idx].add(connection)
        self._next_pattern()
        return connection