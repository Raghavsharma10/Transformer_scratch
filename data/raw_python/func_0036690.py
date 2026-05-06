def get_port_by_name(self, port_name):
        '''Get a port of this component by name.'''
        with self._mutex:
            for p in self.ports:
                if p.name == port_name:
                    return p
            return None