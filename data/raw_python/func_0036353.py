def connections(self):
        '''A list of connections to or from this port.

        This list will be created at the first reference to this property.
        This means that the first reference may be delayed by CORBA calls,
        but others will return quickly (unless a delayed reparse has been
        triggered).

        '''
        with self._mutex:
            if not self._connections:
                self._connections = [Connection(cp, self) \
                                     for cp in self._obj.get_connector_profiles()]
        return self._connections