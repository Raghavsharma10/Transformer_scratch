def _merge_prefix(self, size):
        """Replace the first entries in a deque of strings with a single
        string of up to size bytes.

        >>> d = collections.deque(['abc', 'de', 'fghi', 'j'])
        >>> _merge_prefix(d, 5); print(d)
        deque(['abcde', 'fghi', 'j'])

        Strings will be split as necessary to reach the desired size.
        >>> _merge_prefix(d, 7); print(d)
        deque(['abcdefg', 'hi', 'j'])

        >>> _merge_prefix(d, 3); print(d)
        deque(['abc', 'defg', 'hi', 'j'])

        >>> _merge_prefix(d, 100); print(d)
        deque(['abcdefghij'])
        """
        if len(self._buf) == 1 and len(self._buf[0]) <= size:
            return
        prefix = []
        remaining = size
        while self._buf and remaining > 0:
            chunk = self._buf.popleft()
            if len(chunk) > remaining:
                self._buf.appendleft(chunk[remaining:])
                chunk = chunk[:remaining]
            prefix.append(chunk)
            remaining -= len(chunk)
        if prefix:
            self._buf.appendleft(b''.join(prefix))
        if not self._buf:
            self._buf.appendleft(b'')