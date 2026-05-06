def _tobytes(self):
        """Serializes the write buffer into a single string (bytes).

        Returns:
            a string (bytes) object.
        """
        if not self._has_view:
            # fast path optimization
            if len(self._deque) == 0:
                return b""
            elif len(self._deque) == 1:
                # no copy
                return self._deque[0]
            else:
                return b"".join(self._deque)
        else:
            tmp = [x.tobytes() if isinstance(x, memoryview) else x
                   for x in self._deque]
            return b"".join(tmp)