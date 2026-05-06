def update(self, content):
        """Enumerates the bytes of the supplied bytearray and updates the CRC-64.
           No return value.
        """

        for byte in content:
            self._crc64 = (self._crc64 >> 8) ^ self._lookup_table[(self._crc64 & 0xff) ^ byte]