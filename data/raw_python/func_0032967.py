def read_bytes(self, num_bytes):
        """Reads at most `num_bytes`."""
        buffer_size = len(self._buffer)
        if buffer_size > num_bytes:
            # The buffer is larger than the requested amount of bytes.
            data, self._buffer = self._buffer[:num_bytes], self._buffer[num_bytes:]
        elif 0 < buffer_size <= num_bytes:
            # This might return less bytes than requested.
            data, self._buffer = self._buffer, bytearray()
        else:
            # Buffer is empty. Try to read `num_bytes` and call `read_bytes()`
            # again. This ensures that at most `num_bytes` are returned.
            self._buffer += self.__read__(num_bytes)
            return self.read_bytes(num_bytes)
        return data