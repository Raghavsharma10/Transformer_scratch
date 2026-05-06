def has_port_by_ref(self, port_ref):
        '''Check if this component has a port by the given reference to a CORBA
        PortService object.

        '''
        with self._mutex:
            if self.get_port_by_ref(self, port_ref):
                return True
            return False