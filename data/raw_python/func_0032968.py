def read_until(self, delimiter):
        """Reads until the delimiter is found."""
        if delimiter in self._buffer:
            data, delimiter, self._buffer = self._buffer.partition(delimiter)
            return data
        else:
            self._buffer += self.__read__(self._max_bytes)
            return self.read_until(delimiter)