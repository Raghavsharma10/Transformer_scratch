def rem_callback(self, event, cb):
        '''Remove a callback from this node.

        The callback is removed from the specified event.

        @param cb The callback function to remove.

        '''
        if event not in self._cbs:
            raise exceptions.NoSuchEventError(self.name, event)
        c = [(x[0], x[1]) for x in self._cbs[event]]
        if not c:
            raise exceptions.NoCBError(self.name, event, cb)
        self._cbs[event].remove(c[0])