def write(self, buf, *, start=0, end=None):
        """
        Write the bytes from ``buf`` to the device.

        If ``start`` or ``end`` is provided, then the buffer will be sliced
        as if ``buffer[start:end]``. This will not cause an allocation like
        ``buffer[start:end]`` will so it saves memory.

        :param bytearray buf: buffer containing the bytes to write
        :param int start: Index to start writing from
        :param int end: Index to read up to but not include
        """
        if end is None:
            end = len(buf)
        for i in range(start, end):
            self._writebyte(buf[i])