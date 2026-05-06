def search(self, buf):
        """Search the provided buffer for matching text.

        Search the provided buffer for matching text. If the *match* is found,
        returns a :class:`SequenceMatch` object, otherwise returns ``None``.

        :param buf: Buffer to search for a match.
        :return: :class:`SequenceMatch` if matched, None if no match was found.
        """
        self._check_type(buf)
        normalized = unicodedata.normalize(self.FORM, buf)
        idx = normalized.find(self._text)
        if idx < 0:
            return None
        start = idx
        end = idx + len(self._text)
        return SequenceMatch(self, normalized[start:end], start, end)