def read(self, size = None):
        '''Read at most size bytes, returned as a string.

        If the size argument is negative or omitted, read until EOF is reached.
        Notice that when in non-blocking mode, less data than what was requested
        may be returned, even if no size parameter was given.'''

        if self.closed: raise ValueError('File closed')
        if self._mode in _allowed_write:
            raise Exception('File opened for write only')
        if not self._done_header:
            self._read_header()

        # The encrypted file has been entirely read, so return as much as they want
        # and remove the returned portion from the decrypted buffer
        if self._read_finished:
            if size is None:
                decrypted = self._decrypted_buffer
            else:
                decrypted = self._decrypted_buffer[:size]
            self._decrypted_buffer = self._decrypted[len(decrypted):]
            return decrypted

        # Read everything in one chunk
        if size is None or size < 0:
            self._encrypted_buffer = self._fp.read()
            self._read_finished = True

        else:
            # We fill the encrypted buffer (keeping it with a minimum of 32 bytes in case of the
            # end-of-file checksum) and decrypt into a decrypted buffer 1 block at a time
            while not self._read_finished:

                # We have enough decrypted bytes (or will after decrypting the encrypted buffer)
                available = len(self._decrypted_buffer) + len(self._encrypted_buffer) - 32
                if available >= size: break

                # Read a little extra for the possible final checksum
                data = self._fp.read(BLOCK_SIZE)

                # No data left; we're done
                if not data:
                    self._read_finished = True
                    break

                self._encrypted_buffer += data

        # Decrypt as much of the encrypted data as possible (leaving the final check sum)
        safe = self._encrypted_buffer[:-32]
        self._encrypted_buffer = self._encrypted_buffer[-32:]
        self._decrypted_buffer += self._crypto.decrypt(safe)
        self._checksumer.update(safe)

        # We read all the bytes, only the checksum remains
        if self._read_finished:
            self._check_final_checksum(self._encrypted_buffer)

        # Send back the number of bytes requests and remove them from the buffer
        decrypted = self._decrypted_buffer[:size]
        self._decrypted_buffer = self._decrypted_buffer[size:]

        return decrypted