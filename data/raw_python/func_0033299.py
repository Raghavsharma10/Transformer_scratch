def get_memory(self, start, size):
        """Retrieve an area of memory from IDA.
        Returns a sparse dictionary of address -> value.
        """
        LOG.debug('get_memory: %d bytes from %x', size, start)
        return get_memory(self.ida.idaapi, start, size,
                          default_byte=self.default_byte)