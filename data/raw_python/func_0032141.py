def buffered(self):
        """ Whether write operations should be buffered, i.e. run against a
        local graph before being stored to the main data store. """
        if 'buffered' not in self.config:
            return not isinstance(self.store, (Memory, IOMemory))
        return self.config.get('buffered')