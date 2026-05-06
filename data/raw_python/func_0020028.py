def readinto(self, buf, *, start=0, end=None):
        """
        Read into ``buf`` from the device. The number of bytes read will be the
        length of ``buf``.

        If ``start`` or ``end`` is provided, then the buffer will be sliced
        as if ``buf[start:end]``. This will not cause an allocation like
        ``buf[start:end]`` will so it saves memory.

        :param bytearray buf: buffer to write into
        :param int start: Index to start writing at
        :param int end: Index to write up to but not include
        """
        self._bus.readinto(buf, start=start, end=end)
        if start == 0 and end is None and len(buf) >= 8:
            if self._bus.crc8(buf):
                raise RuntimeError('CRC error.')