def ports(self):
        '''The list of all ports belonging to this component.'''
        with self._mutex:
            if not self._ports:
                self._ports = [ports.parse_port(port, self) \
                               for port in self._obj.get_ports()]
        return self._ports