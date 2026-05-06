def _finalize_write(self):
        'Finishes any unencrypted bytes and writes the final checksum.'

        # Make sure we have written the header
        if not self._done_header:
            self._write_header()

        # Write the remaining decrypted part to disk
        block = self._crypto.encrypt(self._decrypted_buffer)
        self._decrypted = ''
        self._fp.write(block)
        self._checksumer.update(block)

        # Write the final checksum
        self._fp.write(self._checksumer.digest())
        self._valid = True