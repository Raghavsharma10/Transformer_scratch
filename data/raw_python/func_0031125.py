def search(self, buf):
        """Search the provided buffer for a match to the object's regex.

        Search the provided buffer for a match to the object's regex. If the
        *match* is found, returns a :class:`RegexMatch` object, otherwise
        returns ``None``.

        :param buf: Buffer to search for a match.
        :return: :class:`RegexMatch` if matched, None if no match was found.
        """
        match = self._regex.search(self._check_type(buf))
        if match is not None:
            start = match.start()
            end = match.end()
            return RegexMatch(self, buf[start:end], start, end, match.groups())