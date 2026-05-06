def frame_message(self):
        """ Construct a frame around the first complete message in the buffer.
        """
        if self._frame is not None:
            self.discard_message()
        panes = []
        p = origin = self._origin
        extent = self._extent
        while p < extent:
            available = extent - p
            if available < 2:
                break
            chunk_size, = struct_unpack(">H", self._view[p:(p + 2)])
            p += 2
            if chunk_size == 0:
                self._limit = p
                self._frame = MessageFrame(memoryview(self._view[origin:self._limit]), panes)
                return True
            q = p + chunk_size
            panes.append((p - origin, q - origin))
            p = q
        return False