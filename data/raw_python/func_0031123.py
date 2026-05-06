def search(self, buf):
        """Search the provided buffer for matching bytes.

        Search the provided buffer for matching bytes. If the *match* is found,
        returns a :class:`SequenceMatch` object, otherwise returns ``None``.

        :param buf: Buffer to search for a match.
        :return: :class:`SequenceMatch` if matched, None if no match was found.
        """
        idx = self._check_type(buf).find(self._bytes)
        if idx < 0:
            return None
        else:
            start = idx
            end = idx + len(self._bytes)
            return SequenceMatch(self, buf[start:end], start, end)