def _double_prefix(self):
        """Grow the given deque by doubling, but don't split the second chunk just
        because the first one is small.
        """
        new_len = max(len(self._buf[0]) * 2, (len(self._buf[0]) + len(self._buf[1])))
        self._merge_prefix(new_len)