def remove(self, node):
        """Remove a node from the list.

        The *node* argument must be a node that was previously inserted in the
        list
        """
        if node is None or node._prev == -1:
            return
        if node._next is None:
            self._last = node._prev  # last node
        else:
            node._next._prev = node._prev
        if node._prev is None:
            self._first = node._next  # first node
        else:
            node._prev._next = node._next
        node._prev = node._next = -1
        self._size -= 1