def number(self) -> int:
        """Episode number.

        Unique for an anime and episode type, but not unique across episode
        types for the same anime.
        """
        match = self._NUMBER_SUFFIX.search(self.epno)
        return int(match.group(1))