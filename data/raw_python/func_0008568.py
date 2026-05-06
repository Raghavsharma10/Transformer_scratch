def read(self, handle):
        '''Read binary data for this parameter from a file handle.

        This reads exactly enough data from the current position in the file to
        initialize the parameter.
        '''
        self.bytes_per_element, = struct.unpack('b', handle.read(1))
        dims, = struct.unpack('B', handle.read(1))
        self.dimensions = [struct.unpack('B', handle.read(1))[0] for _ in range(dims)]
        self.bytes = b''
        if self.total_bytes:
            self.bytes = handle.read(self.total_bytes)
        size, = struct.unpack('B', handle.read(1))
        self.desc = size and handle.read(size).decode('utf-8') or ''