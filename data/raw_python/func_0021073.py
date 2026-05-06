def sort(self, key=None, reverse=False):
        """
        Sort the items of this collection according to the optional callable
        *key*. If *reverse* is set then the sort order is reversed.

        .. note::
            This sort requires all items to be retrieved from Redis and stored
            in memory.
         """
        def sort_trans(pipe):
            values = list(self.__iter__(pipe))
            values.sort(key=key, reverse=reverse)

            pipe.multi()
            pipe.delete(self.key)
            pipe.rpush(self.key, *(self._pickle(v) for v in values))

            if self.writeback:
                self.cache = {}

        return self._transaction(sort_trans)