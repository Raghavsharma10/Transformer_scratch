def make_connection(self):
        """Create a new connection"""
        if self._created_connections[self._pattern_idx] >= self.max_connections_per_pattern:
            raise ConnectionError("Too many connections")
        self._created_connections[self._pattern_idx] += 1
        conn = self.connection_class(**self.patterns[self._pattern_idx])
        conn._pattern_idx = self._pattern_idx
        return conn