def has_port(self, port):
        '''Return True if this connection involves the given Port object.

        @param port The Port object to search for in this connection's ports.

        '''
        with self._mutex:
            for p in self.ports:
                if not p[1]:
                    # Port owner not in tree, so unknown
                    continue
                if port.object._is_equivalent(p[1].object):
                    return True
            return False