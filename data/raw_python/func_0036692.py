def has_port_by_name(self, port_name):
        '''Check if this component has a port by the given name.'''
        with self._mutex:
            if self.get_port_by_name(port_name):
                return True
            return False