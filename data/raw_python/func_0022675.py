def _edge_in_front(self, edge):
        """ Return the index where *edge* appears in the current front.
        If the edge is not in the front, return -1
        """
        e = (list(edge), list(edge)[::-1])
        for i in range(len(self._front)-1):
            if self._front[i:i+2] in e:
                return i
        return -1