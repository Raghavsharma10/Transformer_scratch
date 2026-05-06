def get_connections_by_dests(self, dests):
        '''Search for all connections involving this and all other ports.'''
        with self._mutex:
            res = []
            for c in self.connections:
                if not c.has_port(self):
                    continue
                has_dest = False
                for d in dests:
                    if c.has_port(d):
                        has_dest = True
                        break
                if has_dest:
                    res.append(c)
            return res