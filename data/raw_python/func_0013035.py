def __fill_buffer(self, size=0):
    """Fills the internal buffer.

    Args:
      size: Number of bytes to read. Will be clamped to
        [self.__buffer_size, MAX_BLOB_FETCH_SIZE].
    """
    read_size = min(max(size, self.__buffer_size), MAX_BLOB_FETCH_SIZE)

    self.__buffer = fetch_data(self.__blob_key, self.__position,
                               self.__position + read_size - 1)
    self.__buffer_position = 0
    self.__eof = len(self.__buffer) < read_size