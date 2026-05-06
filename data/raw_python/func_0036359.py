def disconnect(self):
        '''Disconnect this connection.'''
        with self._mutex:
            if not self.ports:
                raise exceptions.NotConnectedError
            # Some of the connection participants may not be in the tree,
            # causing the port search in self.ports to return ('Unknown', None)
            # for those participants. Search the list to find the first
            # participant that is in the tree (there must be at least one).
            p = self.ports[0][1]
            ii = 1
            while not p and ii < len(self.ports):
                p = self.ports[ii][1]
                ii += 1
            if not p:
                raise exceptions.UnknownConnectionOwnerError
            p.object.disconnect(self.id)