def readline(self, size = None):
        '''Next line from the decrypted file, as a string.

        Retain newline.  A non-negative size argument limits the maximum
        number of bytes to return (an incomplete line may be returned then).
        Return an empty string at EOF.'''

        if self.closed: raise ValueError('file closed')
        if self._mode in _allowed_write:
            raise Exception('file opened for write only')
        if self._read_finished: return None

        line = b''
        while not line.endswith(b'\n') and not self._read_finished and (size is None or len(line) <= size):
            line += self.read(1)
        return line