def _normalize_index(self, index, pipe=None):
        """Convert negative indexes into their positive equivalents."""
        pipe = self.redis if pipe is None else pipe
        len_self = self.__len__(pipe)
        positive_index = index if index >= 0 else len_self + index

        return len_self, positive_index