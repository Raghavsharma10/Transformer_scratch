def index(self, value, start=None, stop=None):
        """
        Return the index of the first occurence of *value*.
        If *start* or *stop* are provided, return the smallest
        index such that ``s[index] == value`` and ``start <= index < stop``.
        """
        def index_trans(pipe):
            len_self, normal_start = self._normalize_index(start or 0, pipe)
            __, normal_stop = self._normalize_index(stop or len_self, pipe)
            for i, v in enumerate(self.__iter__(pipe=pipe)):
                if v == value:
                    if i < normal_start:
                        continue
                    if i >= normal_stop:
                        break
                    return i
            raise ValueError

        return self._transaction(index_trans)