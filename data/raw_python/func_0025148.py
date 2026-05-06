def pop(self, key=__marker, default=__marker):
        """
        If ``key`` is in the pqdict, remove it and return its priority value,
        else return ``default``. If ``default`` is not provided and ``key`` is
        not in the pqdict, raise a ``KeyError``.

        If ``key`` is not provided, remove the top item and return its key, or
        raise ``KeyError`` if the pqdict is empty.

        """
        heap = self._heap
        position = self._position
        # pq semantics: remove and return top *key* (value is discarded)
        if key is self.__marker:
            if not heap:
                raise KeyError('pqdict is empty')
            key = heap[0].key
            del self[key]
            return key
        # dict semantics: remove and return *value* mapped from key
        try:
            pos = position.pop(key)  # raises KeyError
        except KeyError:
            if default is self.__marker:
                raise
            return default
        else:
            node_to_delete = heap[pos]
            end = heap.pop()
            if end is not node_to_delete:
                heap[pos] = end
                position[end.key] = pos
                self._reheapify(pos)
            value = node_to_delete.value
            del node_to_delete
            return value