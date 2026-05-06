def sync_close(self):
        """
        同步关闭
        """
        if self._closed:
            return
        while self._free:
            conn = self._free.popleft()
            if not conn.closed:
                # pragma: no cover
                conn.sync_close()
        for conn in self._used:
            if not conn.closed:
                # pragma: no cover
                conn.sync_close()
            self._terminated.add(conn)
        self._used.clear()
        self._closed = True