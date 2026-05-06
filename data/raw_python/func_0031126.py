def search(self, buf):
        """Search the provided buffer for a match to any sub-searchers.

        Search the provided buffer for a match to any of this collection's
        sub-searchers. If a single matching sub-searcher is found, returns that
        sub-searcher's *match* object. If multiple matches are found, the match
        with the smallest index is returned. If no matches are found, returns
        ``None``.

        :param buf: Buffer to search for a match.
        :return: :class:`RegexMatch` if matched, None if no match was found.
        """
        self._check_type(buf)
        best_match = None
        best_index = sys.maxsize
        for searcher in self:
            match = searcher.search(buf)
            if match and match.start < best_index:
                best_match = match
                best_index = match.start
        return best_match