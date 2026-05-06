def write(self, str):
        '''Write string str to the underlying file.

        Note that due to buffering, flush() or close() may be needed before
        the file on disk reflects the data written.'''

        if self.closed: raise ValueError('File closed')
        if self._mode in _allowed_read:
            raise Exception('File opened for read only')

        if self._valid is not None:
            raise Exception('file already finalized')

        if not self._done_header:
            self._write_header()

        # Encrypt and write the data
        encrypted = self._crypto.encrypt(str)
        self._checksumer.update(encrypted)
        self._fp.write(encrypted)