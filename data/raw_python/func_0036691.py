def get_port_by_ref(self, port_ref):
        '''Get a port of this component by reference to a CORBA PortService
        object.

        '''
        with self._mutex:
            for p in self.ports:
                if p.object._is_equivalent(port_ref):
                    return p
            return None