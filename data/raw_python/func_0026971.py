def wait_closed(self):
        """
        Wait for closing all pool's connections.
        """
        if self._closed:
            return
        if not self._closing:
            raise RuntimeError(
                ".wait_closed() should be called "
                "after .close()"
            )

        while self._free:
            conn = self._free.popleft()
            if not conn.closed:
                yield from conn.close()
            else:
                # pragma: no cover
                pass
        with (yield from self._cond):
            while self.size > self.freesize:
                yield from self._cond.wait()
        self._used.clear()
        self._closed = True