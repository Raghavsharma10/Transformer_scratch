def all_connections(self):
        """Returns a generator over all current connection objects"""
        for i in _xrange(self.num_patterns):
            for c in self._available_connections[i]:
                yield c
            for c in self._in_use_connections[i]:
                yield c