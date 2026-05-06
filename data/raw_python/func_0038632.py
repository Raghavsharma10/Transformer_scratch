def items(self, start=None, stop=None):
        """Return an iterator yielding pairs.

        If *start* is specified, iteration starts at the first pair with a key
        that is larger than or equal to *start*. If not specified, iteration
        starts at the first pair in the list.

        If *stop* is specified, iteration stops at the last pair that is
        smaller than *stop*. If not specified, iteration end with the last pair
        in the list.
        """
        if start is None:
            node = self._head[2]
        else:
            self._find_lt(start)
            node = self._path[0][2]
        while node is not self._tail and (stop is None or node[0] < stop):
            yield (node[0], node[1])
            node = node[2]